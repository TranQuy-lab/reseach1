"""Export local sklearn forests to numerical arrays for safe inference."""
import numpy as np


def save_forest(model, path):
    arrays = {'classes': model.classes_}
    for i, estimator in enumerate(model.estimators_):
        t = estimator.tree_
        for name, values in dict(left=t.children_left, right=t.children_right,
                                 feature=t.feature, threshold=t.threshold, value=t.value[:, 0, :]).items():
            arrays[f'{i}_{name}'] = values
    arrays['n_trees'] = np.array(len(model.estimators_))
    np.savez_compressed(path, **arrays)


def predict_forest(path, x):
    # sklearn trees compare float32 samples to float64 thresholds.
    x = np.asarray(x, dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        result = np.zeros((len(x), len(data['classes'])), dtype=np.float64)
        n_trees = int(data['n_trees'])
        for i in range(n_trees):
            left, right = data[f'{i}_left'], data[f'{i}_right']
            feature, threshold = data[f'{i}_feature'], data[f'{i}_threshold']
            nodes = np.zeros(len(x), dtype=np.int64)
            active = np.flatnonzero(left[nodes] != -1)
            while len(active):
                nd = nodes[active]
                go_left = x[active, feature[nd]] <= threshold[nd]
                nodes[active] = np.where(go_left, left[nd], right[nd])
                active = active[left[nodes[active]] != -1]
            values = data[f'{i}_value'][nodes]
            result += values / values.sum(axis=1, keepdims=True)
    return result / n_trees
