from .base_env import BaseCorridorEnv, EnvObservation, EnvStepResult
from .mock_env import MockCorridorEnv

__all__ = ["BaseCorridorEnv", "EnvObservation", "EnvStepResult", "MockCorridorEnv"]

try:
    from .airsim_env import AirSimCorridorEnv  # noqa: F401
    __all__.append("AirSimCorridorEnv")
except ImportError:
    # The `airsim` package and a running Unreal Engine instance are only
    # required if you actually select env.name == "airsim" in the config.
    pass
