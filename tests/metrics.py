import numpy as np
import pytest

from src.metrics import evaluate_predictions, lift_over_baseline


def test_rank_position_controls_hit_ratio():
    y = np.array([[0, 0, 1, 0, 0],
                  [1, 0, 0, 0, 0]], dtype=np.float32)
    s = np.array([[0.0, 0.9, 0.1, 0.2, 0.3],
                  [0.0, 0.5, 0.4, 0.3, 0.2]], dtype=np.float32)
    res = evaluate_predictions(s, y, k_values=(2, 3, 4))
    assert res[2]['HitRatio'] == 0.0
    assert res[3]['HitRatio'] == 0.0
    assert res[4]['HitRatio'] == 1.0
    assert res[4]['Recall'] == 1.0


def test_rare_only_users_are_excluded():
    y = np.array([[0, 0, 1, 0, 0],
                  [1, 0, 0, 0, 0]], dtype=np.float32)
    s = np.zeros((2, 5), dtype=np.float32)
    res = evaluate_predictions(s, y, k_values=(4,))
    assert res[4]['evaluated_users'] == 1
    assert res[4]['total_users'] == 2


def test_ties_are_broken_by_item_index():
    y = np.array([[0, 0, 1, 0, 0]], dtype=np.float32)
    s = np.zeros((1, 5), dtype=np.float32)
    res = evaluate_predictions(s, y, k_values=(2,))
    assert res[2]['HitRatio'] == 1.0


def test_recall_with_multiple_true_items():
    y = np.array([[0, 1, 1, 0, 0]], dtype=np.float32)
    s = np.array([[0.0, 0.9, 0.0, 0.8, 0.1]], dtype=np.float32)
    res = evaluate_predictions(s, y, k_values=(1, 2))
    assert res[1]['Recall'] == pytest.approx(0.5)
    assert res[2]['Recall'] == pytest.approx(0.5)


def test_lift_over_baseline():
    model = {10: {'HitRatio': 0.6}}
    base = {10: {'HitRatio': 0.5}}
    assert lift_over_baseline(model, base)[10] == pytest.approx(0.2)


def test_rejects_non_binary_targets():
    y = np.array([[0, 2, 0]], dtype=np.float32)
    s = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        evaluate_predictions(s, y, k_values=(1,))
