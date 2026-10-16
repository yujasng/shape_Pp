"""장면, action, 관측, 초기화 설정."""

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import scene_objects_cfg


@configclass
class ShapeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=2500.0,
        ),
    )

    robot: ArticulationCfg = scene_objects_cfg.robot_cfg
    table: AssetBaseCfg = scene_objects_cfg.table_cfg
    cube: RigidObjectCfg = scene_objects_cfg.cube_cfg

    # 실행 옵션에 따라 scene_objects_cfg에서 연결.
    right_wrist_camera: CameraCfg | None = None
    left_wrist_camera: CameraCfg | None = None
    head_camera: CameraCfg | None = None


@configclass
class ActionsCfg:
    joint_action = mdp.JointPositionActionCfg(
        asset_name="robot",

        # mimic 종동 관절은 직접 명령하지 않음.
        joint_names=[
            "lift_joint",
            "arm_[lr]_joint[1-7]",
            "gripper_[lr]_joint1",
            "head_joint[1-2]",
        ],

        scale=1.0,

        # 목표 위치 = 초기 관절 위치 + action.
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # 영상은 현재 __init__.py의 camera_outputs에서 별도로 읽음.
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class ShapeEnvCfg(ManagerBasedEnvCfg):
    scene: ShapeSceneCfg = ShapeSceneCfg(
        num_envs=1,
        env_spacing=4.0,
    )

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 물리 120Hz, 환경 제어 60Hz.
        self.decimation = 2
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        # 사용자 화면의 관찰용 시점.
        # 로봇에 부착된 센서 카메라와 별개.
        self.viewer.eye = (2.8, 2.8, 2.2)
        self.viewer.lookat = (0.6, 0.0, 0.8)
        self.viewer.resolution = (1280, 720)


def get_ground_truth(env):
    """평가기 전용 cube 상태. quaternion 순서는 wxyz."""

    cube = env.scene["cube"]

    return {
        "cube": {
            "position_w": cube.data.root_pos_w.clone(),
            "quaternion_wxyz": cube.data.root_quat_w.clone(),
            "linear_velocity_w": cube.data.root_lin_vel_w.clone(),
        }
    }