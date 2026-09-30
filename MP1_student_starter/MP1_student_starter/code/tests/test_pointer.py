"""Checks for the within-window pointer cache. Run: python -m unittest discover -s tests -v (CPU)."""
import unittest
import torch
from student import build_model


def make(pointer=True):
    torch.manual_seed(17)
    # ptr_null_bias=0 so the pointer carries real probability mass in these tests.
    config = dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                  pointer=pointer, ptr_dim=16, ptr_null_bias=0.0)
    return build_model(config).eval()


def tokens(batch=2, length=16):
    return torch.randint(0, 30, (batch, length))  # small alphabet, so tokens repeat and the cache is used


class PointerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        self.model = make()

    def test_future_inputs_cannot_change_earlier_predictions(self):
        x = tokens()
        changed = x.clone(); changed[:, 9:] = (changed[:, 9:] + 19) % 2048
        with torch.no_grad():
            a, b = self.model.predict_log_probs(x), self.model.predict_log_probs(changed)
        torch.testing.assert_close(a[:, :9], b[:, :9], atol=1e-6, rtol=1e-6)
        self.assertGreater((a[:, 9:] - b[:, 9:]).abs().max().item(), 1e-4)

    def test_normalized_and_examples_independent(self):
        x = tokens()
        with torch.no_grad():
            together = self.model.predict_log_probs(x)
            alone = self.model.predict_log_probs(x[:1])
        self.assertEqual(tuple(together.shape), (2, 16, 2048))
        self.assertTrue(torch.isfinite(together).all())
        torch.testing.assert_close(together.logsumexp(-1), torch.zeros(2, 16), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-5)

    def test_state_resets_between_windows(self):
        x = tokens(1, 12)
        with torch.no_grad():
            first = self.model.predict_log_probs(x)
            self.model.predict_log_probs((x + 7) % 2048)
            again = self.model.predict_log_probs(x)
        torch.testing.assert_close(first, again, atol=1e-6, rtol=1e-6)

    def test_pointer_changes_output_and_receives_gradients(self):
        x = tokens(2, 17)
        plain = make(pointer=False)
        with torch.no_grad():
            self.assertGreater((self.model.predict_log_probs(x) - plain.predict_log_probs(x)).abs().max().item(), 1e-4)
        loss = torch.nn.functional.cross_entropy(self.model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for layer in (self.model.ptr_q, self.model.ptr_k, self.model.ptr_null):
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_pointer_off_by_default(self):
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2, context=256))
        self.assertFalse(model.pointer)
        self.assertFalse(any(name.startswith('ptr_') for name, _ in model.named_parameters()))


if __name__ == '__main__':
    unittest.main()
