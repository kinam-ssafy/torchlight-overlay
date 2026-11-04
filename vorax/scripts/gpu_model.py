"""Export numeric XGBoost regressors to Vorax's existing CPU-only tree format."""
import json
import numpy as np


def split_boundary(value):
    """x64 < boundary iff float32(x64) < value (round-to-nearest-even).

    Training uses float32 features, while the production engine uses float64.
    Moving each boundary avoids changing the production feature extractor.
    """
    threshold = np.float32(value)
    previous = np.nextafter(threshold, np.float32(-np.inf))
    if not np.isfinite(threshold) or not np.isfinite(previous):
        raise ValueError('non-finite split boundary')
    midpoint = (float(previous) + float(threshold)) / 2
    # An odd threshold loses the midpoint tie to its even predecessor.
    if threshold.view(np.uint32) & 1:
        midpoint = float(np.nextafter(midpoint, np.inf))
    return midpoint


def export_model(booster):
    raw = json.loads(booster.save_raw(raw_format='json'))['learner']
    if raw['objective']['name'] != 'reg:squarederror':
        raise ValueError('only squared-error regression is supported')
    params = raw['learner_model_param']
    if int(params['num_target']) != 1 or int(params['num_class']) != 0:
        raise ValueError('only single-output regression is supported')
    base = json.loads(params['base_score'])
    if isinstance(base, list):
        if len(base) != 1:
            raise ValueError('multiple base scores')
        base = base[0]
    gb = raw['gradient_booster']
    if gb['name'] != 'gbtree':
        raise ValueError('only gbtree is supported')
    trees = []
    for source in gb['model']['trees']:
        if any(source['split_type']) or source['categories']:
            raise ValueError('categorical splits are not supported')
        tree = {key: [] for key in ('f', 't', 'l', 'r', 'v')}
        for i, left in enumerate(source['left_children']):
            leaf = left == -1
            value = source['split_conditions'][i]
            tree['f'].append(-1 if leaf else source['split_indices'][i])
            tree['t'].append(0.0 if leaf else split_boundary(value))
            tree['l'].append(0 if leaf else left)
            tree['r'].append(0 if leaf else source['right_children'][i])
            tree['v'].append(float(np.float32(value)) if leaf else 0.0)
        trees.append(tree)
    return {'base': float(np.float32(base)), 'trees': trees}


def predict_export(model, features):
    """Validation only; the shipped executable uses Rust, not this interpreter."""
    x = np.asarray(features, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Vorax features must be finite')
    result = np.full(len(x), model['base'], dtype=np.float64)
    for tree in model['trees']:
        nodes = np.zeros(len(x), dtype=np.int64)
        f, t, l, r, v = (np.asarray(tree[k]) for k in ('f', 't', 'l', 'r', 'v'))
        active = f[nodes] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            at = nodes[rows]
            nodes[rows] = np.where(x[rows, f[at]] < t[at], l[at], r[at])
            active = f[nodes] >= 0
        result += v[nodes]
    return result
