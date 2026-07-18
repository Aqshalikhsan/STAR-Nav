"""Pure-Python sanity tests for world_gen.py -- no ROS 2, Gazebo, or MAVLink
install required. Run with:

    pytest ros_gazebo_bridge/test/test_world_gen.py -v
"""
import os
import sys

_BRIDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_BRIDGE_DIR)
sys.path.insert(0, _REPO_ROOT)    # for `import star_nav`
sys.path.insert(0, _BRIDGE_DIR)   # for `import ros_gazebo_bridge` (colcon-package layout, see setup.py)

import numpy as np

from ros_gazebo_bridge.world_gen import build_corridor_world, load_world_layout, to_sdf, write_world_bundle
from star_nav.utils.config import load_config


def _scenario_cfg(scenario="A"):
    cfg = load_config()
    return cfg.env.scenarios[scenario]


def test_build_corridor_world_matches_mock_env_algorithm():
    from star_nav.envs.mock_env import MockCorridorEnv

    cfg = load_config()
    mock = MockCorridorEnv(cfg.env)
    mock.rng = np.random.default_rng(123)
    mock_world = mock._generate_world("A")

    world = build_corridor_world(_scenario_cfg("A"), cfg.env.area_size_m, scenario="A", seed=123)

    assert np.allclose(world.tree_xy, mock_world.tree_xy)
    assert world.corridor_center_y == mock_world.corridor_center_y
    assert np.allclose(world.goal_xy, mock_world.goal_xy)


def test_to_sdf_contains_one_model_per_trunk():
    cfg = load_config()
    world = build_corridor_world(_scenario_cfg("A"), cfg.env.area_size_m, scenario="A", seed=1)

    # Default: oil_palm mesh trunks (one <include> per tree, no external
    # asset actually needed just to render this as a string).
    sdf = to_sdf(world)
    assert sdf.count("<name>trunk_") == len(world.tree_xy)
    assert sdf.count("model://oil_palm") == len(world.tree_xy)
    assert "<sdf version=" in sdf
    assert "ground_plane" in sdf

    # Fallback: plain cylinder trunks (no external asset dependency).
    sdf_cyl = to_sdf(world, use_mesh_trunks=False)
    assert sdf_cyl.count("<model name=\"trunk_") == len(world.tree_xy)
    assert "ground_plane" in sdf_cyl


def test_write_and_load_world_bundle_round_trips(tmp_path):
    cfg = load_config()
    world = build_corridor_world(_scenario_cfg("B"), cfg.env.area_size_m, scenario="B", seed=7)

    out_prefix = str(tmp_path / "scenario_b")
    write_world_bundle(world, out_prefix)

    assert os.path.exists(out_prefix + ".sdf")
    assert os.path.exists(out_prefix + ".world.json")

    loaded = load_world_layout(out_prefix + ".world.json")
    assert np.allclose(loaded.tree_xy, world.tree_xy)
    assert loaded.scenario == "B"
    assert loaded.trunk_radius == world.trunk_radius
    assert np.allclose(loaded.goal_xy, world.goal_xy)


if __name__ == "__main__":
    test_build_corridor_world_matches_mock_env_algorithm()
    test_to_sdf_contains_one_model_per_trunk()
    print("world_gen sanity tests passed (skip tmp_path-based test when run directly).")
