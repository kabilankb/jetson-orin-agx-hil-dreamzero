"""DROID tabletop env for Isaac Lab 3. Import only after isaaclab.app.AppLauncher has started."""
import gymnasium as gym

from .env import SCENES, DroidEnvCfg  # noqa: F401

if "DROID-Lab3" not in gym.registry:
    gym.register(
        id="DROID-Lab3",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": DroidEnvCfg},
        disable_env_checker=True,
    )
