import pytest
from ai_essentials.optimizer import SGD
from ai_essentials.value import Value


def test_sgd_step_updates_parameters():
    p1 = Value(1.0)
    p2 = Value(-2.0)
    p1.grad = 0.5
    p2.grad = -1.0
    opt = SGD([p1, p2], lr=0.1)
    opt.step()
    assert pytest.approx(p1.data, rel=1e-6) == 1.0 - 0.1 * 0.5
    assert pytest.approx(p2.data, rel=1e-6) == -2.0 - 0.1 * -1.0


def test_sgd_zero_grad_sets_all_grads_to_zero():
    p1 = Value(1.0)
    p2 = Value(-2.0)
    p1.grad = 0.5
    p2.grad = -1.0
    opt = SGD([p1, p2], lr=0.1)
    opt.zero_grad()
    assert p1.grad == 0.0
    assert p2.grad == 0.0


def test_sgd_step_and_zero_grad_work_together():
    p = Value(2.0)
    p.grad = 1.0
    opt = SGD([p], lr=0.2)
    opt.step()
    opt.zero_grad()
    assert pytest.approx(p.data, rel=1e-6) == 2.0 - 0.2 * 1.0
    assert p.grad == 0.0
