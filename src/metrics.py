import argparse
import json
import os

import numpy as np

RARE_IDX = 0
DEFAULT_K_VALUES = (1, 3, 5, 9)


def validate_inputs(scores, y_true):
    if scores.ndim != 2 or y_true.ndim != 2:
        raise ValueError('scores and y_true must be 2D (users x items)')
    if scores.shape != y_true.shape:
        raise ValueError(f'shape mismatch: scores {scores.shape} vs y_true {y_true.shape}')
    if not np.all(np.isfinite(scores)):
        raise ValueError('scores contain NaN or Inf')
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError('y_true must be binary')


def hit_matrix_top_k(scores, y_true, max_k):
    validate_inputs(scores, y_true)
    cand = scores[:, RARE_IDX + 1:]
    truth = y_true[:, RARE_IDX + 1:] > 0
    if max_k > cand.shape[1]:
        raise ValueError(f'max_k={max_k} exceeds number of real items {cand.shape[1]}')
    n_true = truth.sum(axis=1)
    valid = n_true > 0
    order = np.argsort(-cand[valid], axis=1, kind='stable')[:, :max_k]
    hits = np.take_along_axis(truth[valid], order, axis=1).astype(np.int32)
    return hits, n_true[valid], valid


def hit_ratio_at_k(hits, k):
    return float(np.mean(hits[:, :k].sum(axis=1) > 0))


def recall_at_k(hits, n_true, k):
    return float(np.mean(hits[:, :k].sum(axis=1) / n_true))


def evaluate_predictions(scores, y_true, k_values=DEFAULT_K_VALUES):
    ks = sorted({int(k) for k in k_values})
    if ks[0] < 1:
        raise ValueError('K values must be positive')
    hits, n_true, valid = hit_matrix_top_k(scores, y_true, ks[-1])
    results = {}
    for k in ks:
        results[k] = {
            'HitRatio': hit_ratio_at_k(hits, k),
            'Recall': recall_at_k(hits, n_true, k),
            'evaluated_users': int(valid.sum()),
            'total_users': int(valid.size),
        }
    return results


def lift_over_baseline(model_results, baseline_results, metric='HitRatio'):
    lifts = {}
    for k in model_results:
        base = baseline_results[k][metric]
        lifts[k] = (model_results[k][metric] - base) / max(base, 1e-12)
    return lifts


def parse_named_paths(items):
    pairs = []
    for item in items:
        if '=' not in item:
            raise ValueError(f"expected name=path, got '{item}'")
        name, path = item.split('=', 1)
        pairs.append((name, path))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compute Hit-Ratio@K and Recall@K for saved score matrices')
    parser.add_argument('--targets', required=True, help='path to y_target.npy (users x items, binary)')
    parser.add_argument('--scores', nargs='+', required=True, help='one or more name=path entries of score matrices')
    parser.add_argument('--k', nargs='+', type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument('--baseline', default=None, help='name of the scores entry used as the lift reference')
    parser.add_argument('--output', default=None, help='optional path for a JSON report')
    args = parser.parse_args(argv)

    y_true = np.load(args.targets)
    all_results = {}
    for name, path in parse_named_paths(args.scores):
        scores = np.load(path)
        all_results[name] = evaluate_predictions(scores, y_true, args.k)

    header = f"{'Model':<20}{'K':>5}{'HitRatio':>11}{'Recall':>10}"
    print(header)
    print('-' * len(header))
    for name, res in all_results.items():
        for k, r in res.items():
            print(f"{name:<20}{k:>5}{r['HitRatio']:>11.4f}{r['Recall']:>10.4f}")

    lifts = {}
    if args.baseline:
        if args.baseline not in all_results:
            raise ValueError(f"baseline '{args.baseline}' not among scores entries")
        base_res = all_results[args.baseline]
        for name, res in all_results.items():
            if name == args.baseline:
                continue
            lifts[name] = {
                'HitRatio': lift_over_baseline(res, base_res, 'HitRatio'),
                'Recall': lift_over_baseline(res, base_res, 'Recall'),
            }
            for k in res:
                print(f"{name} vs {args.baseline} @K={k}: "
                      f"HitRatio lift {lifts[name]['HitRatio'][k]:+.2%}, "
                      f"Recall lift {lifts[name]['Recall'][k]:+.2%}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        report = {
            'results': {name: {str(k): v for k, v in res.items()} for name, res in all_results.items()},
            'lift': {name: {m: {str(k): v for k, v in d.items()} for m, d in lf.items()} for name, lf in lifts.items()},
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'Report written to {args.output}')

    return all_results


if __name__ == '__main__':
    main()
