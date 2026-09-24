"""Regression checks for the revised paper's AGSS projection equations."""

import torch

from star_nav.models.agss_ppo import AGSSShield, ComplexityHead


def _shield(tau: float = 1.0) -> AGSSShield:
    # A supplied head makes c_t = sigmoid(0) = 0.5, independent of h_t.
    weights = (torch.zeros(2), torch.zeros(1))
    return AGSSShield(
        d0=0.60,
        alpha=0.35,
        complexity_dim=2,
        device=torch.device("cpu"),
        tau=tau,
        complexity_weights=weights,
    )


def test_tau_scales_velocity_bounds_without_changing_the_margin():
    action = torch.tensor([[0.2, 1.5, 0.0, 0.0]])
    belief = torch.zeros((1, 2))
    left, right = torch.tensor([1.0]), torch.tensor([2.0])
    one_second = _shield(1.0).project(action, belief, left, right)
    two_seconds = _shield(2.0).project(action, belief, left, right)

    assert torch.allclose(one_second["d_safe"], torch.tensor([0.775]))
    assert torch.allclose(one_second["v_y_min"], torch.tensor([-0.225]))
    assert torch.allclose(one_second["v_y_max"], torch.tensor([1.225]))
    assert torch.allclose(two_seconds["v_y_min"], torch.tensor([-0.1125]))
    assert torch.allclose(two_seconds["v_y_max"], torch.tensor([0.6125]))
    assert torch.allclose(two_seconds["safe_action"][:, 1], torch.tensor([0.6125]))


def test_empty_interval_uses_zero_lateral_velocity_and_preserves_raw_bounds():
    action = torch.tensor([[0.2, -0.4, 0.1, 0.0]])
    result = _shield().project(
        action, torch.zeros((1, 2)), torch.tensor([0.70]), torch.tensor([0.70])
    )

    assert result["collapsed"].item()
    assert torch.allclose(result["v_y_min"], torch.tensor([0.075]))
    assert torch.allclose(result["v_y_max"], torch.tensor([-0.075]))
    assert torch.allclose(result["safe_action"][:, 1], torch.zeros(1))
    assert torch.allclose(result["safe_action"][:, [0, 2, 3]], action[:, [0, 2, 3]])


def test_normalized_action_is_projected_in_meters_per_second():
    shield = AGSSShield(0.6, 0.35, 2, torch.device("cpu"),
                       complexity_weights=(torch.zeros(2), torch.zeros(1)),
                       lateral_action_scale=2.5)
    out = shield.project(torch.tensor([[0.0, 0.8, 0.0, 0.0]]),
                         torch.zeros((1, 2)), torch.tensor([2.0]), torch.tensor([1.275]))
    assert torch.allclose(out["v_y_max"], torch.tensor([0.5]))
    assert torch.allclose(out["safe_action"][:, 1], torch.tensor([0.2]))
    assert torch.allclose(out["correction_magnitude"], torch.tensor([1.5]))


def test_complexity_head_target_and_stopped_belief_gradient():
    head = ComplexityHead(2, w_ref=8.0)
    belief = torch.ones((1, 2), requires_grad=True)
    target = head.target(torch.tensor([2.0]), torch.tensor([2.0]))
    assert torch.allclose(target, torch.tensor([0.5]))
    head.loss(belief, torch.tensor([2.0]), torch.tensor([2.0])).backward()
    assert belief.grad is None
    assert head.linear.weight.grad is not None
