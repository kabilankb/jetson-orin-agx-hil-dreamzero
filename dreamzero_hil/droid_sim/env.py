"""DROID tabletop env ported to Isaac Lab 3.

Derived from droid_environment.py in https://github.com/arhanjain/sim-evals (Isaac Lab 2.2), which
DreamZero uses for its DROID sim evaluation. sim-evals carries no license file; the scene USDs are not
redistributed here (scripts/download_assets.sh fetches them from HF owhan/DROID-sim-environments).

Changes from the Isaac Lab 2.2 original:
- quaternions (camera offsets, scene objects) converted from (w, x, y, z) to (x, y, z, w)
- ProxyArray joint data read via .torch
- PhysX tuning goes through isaaclab_physx PhysxCfg (sim.physics)
Cameras, robot, actions, timing (15 Hz control, 120 Hz physics) are unchanged.
"""
from pathlib import Path

import numpy as np
import torch
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, noise
from isaaclab_physx.physics import PhysxCfg
from pxr import Usd, UsdPhysics

from .robot import ASSET_PATH, DROID_ROBOT

SCENES = {
    1: "put the cube in the bowl",
    2: "pick up the can and put it in the mug",
    3: "put the banana in the bin",
}


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def _cam(prim_path, focal_length, pos, rot_xyzw):
    return CameraCfg(
        prim_path=prim_path,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot_xyzw, convention="opengl"),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = DROID_ROBOT

    # original (w,x,y,z) values: (-0.393, -0.195, 0.399, 0.805), (0.805, 0.399, -0.195, -0.393),
    # (-0.420, 0.570, 0.576, -0.409)
    external_cam = _cam("{ENV_REGEX_NS}/external_cam", 2.1, (0.05, 0.57, 0.66), (-0.195, 0.399, 0.805, -0.393))
    external_cam_2 = _cam("{ENV_REGEX_NS}/external_cam_2", 2.1, (0.05, -0.57, 0.66), (0.399, -0.195, -0.393, 0.805))
    wrist_cam = _cam(
        "{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        2.8,
        (0.011, -0.031, -0.074),
        (0.570, 0.576, -0.409, -0.420),
    )

    def dynamic_scene(self, scene_name):
        environment_path = ASSET_PATH / f"scene{scene_name}.usd"
        self.scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(usd_path=str(environment_path)),
        )
        stage = Usd.Stage.Open(str(environment_path))
        for child in stage.GetPrimAtPath("/World").GetChildren():
            if not UsdPhysics.RigidBodyAPI(child):
                continue
            name = child.GetName()
            pos = child.GetAttribute("xformOp:translate").Get()
            q = child.GetAttribute("xformOp:orient").Get()
            im = q.GetImaginary()
            setattr(
                self,
                name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(pos), rot=(im[0], im[1], im[2], q.GetReal())
                    ),
                ),
            )


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    """Gripper: action > 0.5 closes (DROID convention), else opens."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if actions.dtype == torch.bool:
            binary_mask = actions == 0
        else:
            binary_mask = actions > 0.5
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    class_type = BinaryJointPositionZeroToOneAction


@configclass
class ActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )
    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    idx = [robot.data.joint_names.index(n) for n in ARM_JOINTS]
    return _t(robot.data.joint_pos)[:, idx]


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    idx = [robot.data.joint_names.index("finger_joint")]
    return _t(robot.data.joint_pos)[:, idx] / (np.pi / 4)


def _img(name):
    return ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg(name), "data_type": "rgb", "normalize": False},
    )


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1))
        external_cam = _img("external_cam")
        external_cam_2 = _img("external_cam_2")
        wrist_cam = _img("wrist_cam")

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _Empty:
    pass


@configclass
class DroidEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)
    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = _Empty()
    terminations = TerminationsCfg()
    commands = _Empty()
    events = EventCfg()
    curriculum = _Empty()

    def __post_init__(self):
        self.episode_length_s = 30
        self.viewer.eye = (1.6, 0.0, 1.2)
        self.viewer.lookat = (0.4, 0.0, 0.2)
        self.decimation = 8
        self.sim.dt = 1 / (15 * 8)
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            enable_ccd=True,
            gpu_temp_buffer_capacity=2**30,
            gpu_heap_capacity=2**30,
            gpu_collision_stack_size=2**30,
        )
        self.rerender_on_reset = True

    def set_scene(self, scene_name):
        self.scene.dynamic_scene(scene_name)
