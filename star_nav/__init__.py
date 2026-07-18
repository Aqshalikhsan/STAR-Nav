"""STAR-Nav: Spatio-Temporal Adaptive Reinforcement Learning for
autonomous monocular UAV navigation in GPS-denied, perceptually
repetitive plantation corridors.

Package layout mirrors the paper's Section 3 structure:

    star_nav.models.sacr       -> Structure-Aware Corridor Representation
    star_nav.models.camr       -> Consistency-Aware Memory Representation
    star_nav.models.agss_ppo   -> PPO actor-critic + Adaptive Geometric
                                   Safety Shield
    star_nav.envs              -> Simulation environments (mock + AirSim)
    star_nav.training           -> Three-phase decoupled training pipeline
    star_nav.evaluation          -> Metrics and scenario/weather sweeps
"""

__version__ = "1.0.0"
