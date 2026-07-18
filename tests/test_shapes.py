"""Sanity tests: every module's output shape matches the dimensionality
stated in the paper (d_s, d_d, 2*d_h, action_dim=4, etc.), and the mock
environment satisfies the BaseCorridorEnv contract. Run with:

    pytest tests/test_shapes.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from star_nav.envs.mock_env import MockCorridorEnv
from star_nav.models.agss_ppo import ActorCritic, AGSSShield
from star_nav.models.camr import CAMR, CausalWindowBuffer
from star_nav.models.sacr import SACR, sacr_loss
from star_nav.models.camr import camr_loss
from star_nav.utils.config import load_config

BATCH = 2
IMG_H, IMG_W = 120, 160


def _make_sacr():
    return SACR(feature_channels=32, num_seg_classes=5, geom_dim=4,
                geom_hidden=[64, 32], struct_dim=32, depth_pool_regions=3)


def test_sacr_output_shapes():
    sacr = _make_sacr()
    image = torch.rand(BATCH, 3, IMG_H, IMG_W)
    out = sacr(image, need_seg=True)

    assert out.z_struct_aug.shape == (BATCH, 32 + 3)
    assert out.theta_corr.shape == (BATCH, 4)
    assert out.seg_logits.shape == (BATCH, 5, IMG_H, IMG_W)


def test_sacr_loss_runs_and_backprops():
    sacr = _make_sacr()
    image = torch.rand(BATCH, 3, IMG_H, IMG_W)
    out = sacr(image, need_seg=True)
    seg_target = torch.randint(0, 5, (BATCH, IMG_H, IMG_W))
    theta_gt = torch.rand(BATCH, 4)

    losses = sacr_loss(out, seg_target, theta_gt)
    losses["L_SACR"].backward()
    assert torch.isfinite(losses["L_SACR"])


def test_camr_output_shape_and_causal_window():
    sacr = _make_sacr()
    d_s_aug = sacr.z_struct_aug_dim
    camr = CAMR(z_struct_aug_dim=d_s_aug, pose_dim=7, imu_dim=6, window_size=6, hidden_dim=16)

    window = torch.rand(BATCH, 6, camr.input_dim)
    out = camr(window)
    assert out.h_t.shape == (BATCH, 32)  # 2 * hidden_dim

    predicted_next = camr.predict_next(out.h_t)
    assert predicted_next.shape == (BATCH, d_s_aug)

    target_next = torch.rand(BATCH, d_s_aug)
    losses = camr_loss(out, target_next, predicted_next)
    losses["L_CAMR"].backward()
    assert torch.isfinite(losses["L_CAMR"])


def test_causal_window_buffer_left_pads_and_is_causal():
    buf = CausalWindowBuffer(window_size=4, input_dim=8, device=torch.device("cpu"))
    x0 = torch.rand(1, 8)
    window = buf.push(x0)
    assert window.shape == (1, 4, 8)
    # Early-episode left-padding: all four slots equal the first observation.
    assert torch.allclose(window[0, 0], window[0, -1])

    x1 = torch.rand(1, 8)
    window2 = buf.push(x1)
    assert torch.allclose(window2[0, -1], x1[0])  # newest is always last


def test_actor_critic_and_agss_shield():
    belief_dim = 32
    ac = ActorCritic(belief_dim=belief_dim, action_dim=4, actor_hidden=[32, 16], critic_hidden=[32, 16])
    h_t = torch.rand(BATCH, belief_dim)

    sample = ac.act(h_t)
    assert sample.action.shape == (BATCH, 4)
    assert sample.value.shape == (BATCH,)

    agss = AGSSShield(d0=0.6, alpha=1.2, complexity_dim=belief_dim, device=torch.device("cpu"))
    d_left = torch.full((BATCH,), 2.0)
    d_right = torch.full((BATCH,), 2.0)
    projection = agss.project(sample.action, h_t, d_left, d_right)

    assert projection["safe_action"].shape == (BATCH, 4)
    # Only the lateral component (index 1) may differ from the raw action.
    raw = sample.action
    safe = projection["safe_action"]
    assert torch.allclose(raw[:, [0, 2, 3]], safe[:, [0, 2, 3]])


def test_mock_env_step_contract():
    cfg = load_config()
    env = MockCorridorEnv(cfg.env)
    obs = env.reset(scenario="A", weather="clear_day")
    assert obs.rgb.dtype == np.uint8
    assert obs.pose.shape == (7,)
    assert obs.imu.shape == (6,)

    result = env.step(np.array([0.5, 0.0, 0.0, 0.0]))
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert result.info.seg_mask.shape == obs.rgb.shape[:2]


if __name__ == "__main__":
    test_sacr_output_shapes()
    test_sacr_loss_runs_and_backprops()
    test_camr_output_shape_and_causal_window()
    test_causal_window_buffer_left_pads_and_is_causal()
    test_actor_critic_and_agss_shield()
    test_mock_env_step_contract()
    print("All sanity tests passed.")
