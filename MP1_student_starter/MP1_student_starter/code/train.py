"""Student recipe. With default flags this reproduces the baseline recipe exactly:
1,200 steps x 32 sequences x 256 targets = 9,830,400 processed training targets.

Extra options: learning rate / Adam betas / weight decay / warmup / final-LR ratio, no weight decay on
1-D parameters, and an exponential moving average (EMA) of the weights. EMA is self-trained weight
averaging (allowed). Raw and EMA weights are compared on VALIDATION BPB only at every --eval-every point and the best set is saved.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def learning_rate(step, args):
    warmup = min(1., (step + 1) / args.warmup)
    cosine = args.min_lr_ratio + (1 - args.min_lr_ratio) * .5 * (1 + math.cos(math.pi * step / args.steps))
    return args.lr * warmup * cosine


def validate(candidates, data):
    results = {}
    for name, candidate in candidates.items():
        result = score(candidate, *data['validation'], next(candidate.parameters()).device, 'fp32')
        result.pop('window_nll_nats')
        results[name] = result
    return results


def keep_best(best, name, result, model, step):
    """Remember the weights with the lowest VALIDATION bpb seen so far (raw or EMA, any eval step)."""
    if result['bpb'] < best['bpb']:
        best.update(bpb=result['bpb'], name=name, step=step, result=result,
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    # Recipe options (defaults = baseline recipe)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr-ratio', type=float, default=.1, help='Final LR as a fraction of --lr.')
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--no-decay-1d', action='store_true', help='No weight decay on norms/biases.')
    p.add_argument('--beta1', type=float, default=.9)
    p.add_argument('--beta2', type=float, default=.999)
    p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--ema-decay', type=float, default=0., help='0 disables EMA; e.g. 0.999.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.warmup < 1:
        p.error('Batch size, warmup and step count must be positive.')
    if not 0 <= args.ema_decay < 1:
        p.error('--ema-decay must be in [0, 1).')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if args.no_decay_1d and parameter.ndim < 2 else decay).append(parameter)
    groups = [{'params': decay, 'weight_decay': args.weight_decay}]
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))

    ema = None
    if args.ema_decay > 0:
        ema = copy.deepcopy(model).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        model_params, ema_params = list(model.parameters()), list(ema.parameters())

    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best = {'bpb': float('inf')}
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if ema is not None:
            decay_now = min(args.ema_decay, (step+1)/(step+10))  # warm-up so early weights are not over-weighted
            with torch.no_grad():
                torch._foreach_mul_(ema_params, decay_now)
                torch._foreach_add_(ema_params, model_params, alpha=1-decay_now)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'lr':lr,'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            candidates = {'raw': model} | ({'ema': ema} if ema is not None else {})
            results = validate(candidates, data)
            intermediate_validation_seconds += sum(r['seconds'] for r in results.values())
            for name, r in results.items():
                keep_best(best, name, r, candidates[name], step+1)
            validation_history.append({'step':step+1, **{f'{k}_bpb': r['bpb'] for k, r in results.items()}})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds

    candidates = {'raw': model} | ({'ema': ema} if ema is not None else {})
    final = validate(candidates, data)
    for name, r in final.items():
        keep_best(best, name, r, candidates[name], args.steps)
    selected = f"{best['name']}@{best['step']}"  # best-on-validation weights; the test split is never used
    validation = best['result']
    parameters = sum(p.numel() for p in model.parameters())
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':best['state'],'seed':args.seed,'weights':selected,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    hyper = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':parameters,'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'selected_weights':selected,
              'validation_all':{k: {'bpb': r['bpb'], 'token_ppl': r['token_ppl']} for k, r in final.items()},
              'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'hyperparameters':hyper,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
