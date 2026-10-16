"""로봇, 테이블, cube, 로봇 카메라 설정.

거리: m
회전 관절: rad
Quaternion: (w, x, y, z)
"""

from pathlib import Path

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
)
from isaaclab.sensors import CameraCfg


# -----------------------------------------------------------------------------
# USD 경로
# -----------------------------------------------------------------------------
USD_DIR = Path(__file__).resolve().parent / "usd"

ROBOT_USD_PATH = (USD_DIR / "FFW_BG2.usd").as_posix()
TABLE_USD_PATH = (USD_DIR / "table.usd").as_posix()


# -----------------------------------------------------------------------------
# 배치
# -----------------------------------------------------------------------------
ROBOT_POS = (0.0, 0.0, 0.0)
TABLE_POS = (0.95, 0.0, 0.0)

# 실제 상판의 world Z 좌표. 테이블 크기를 바꾸는 값이 아님.
TABLETOP_Z = 0.75

CUBE_SIZE = (0.05, 0.05, 0.05)

CUBE_POS = (
    TABLE_POS[0] - 0.20,
    TABLE_POS[1],
    TABLETOP_Z + CUBE_SIZE[2] / 2.0 + 0.005,
)


# -----------------------------------------------------------------------------
# 로봇
# -----------------------------------------------------------------------------
robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",

    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD_PATH,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS,
        rot=(1.0, 0.0, 0.0, 0.0),

        # 실제 USD에서 충돌 없는 초기 자세인지 확인.
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    actuators={
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            effort_limit_sim=1e6,
            velocity_limit_sim=0.2,
            stiffness=10000.0,
            damping=100.0,
        ),

        "arm_proximal": ImplicitActuatorCfg(
            joint_names_expr=["arm_[lr]_joint[1-2]"],
            effort_limit_sim=61.4,
            velocity_limit_sim=15.0,
            stiffness=600.0,
            damping=30.0,
        ),

        "arm_middle": ImplicitActuatorCfg(
            joint_names_expr=["arm_[lr]_joint[3-6]"],
            effort_limit_sim=31.7,
            velocity_limit_sim=15.0,
            stiffness=600.0,
            damping=20.0,
        ),

        "arm_distal": ImplicitActuatorCfg(
            joint_names_expr=["arm_[lr]_joint7"],
            effort_limit_sim=5.1,
            velocity_limit_sim=6.0,
            stiffness=200.0,
            damping=3.0,
        ),

        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_[lr]_joint1"],
            effort_limit_sim=30.0,
            velocity_limit_sim=2.2,
            stiffness=100.0,
            damping=4.0,
        ),

        "gripper_follower": ImplicitActuatorCfg(
            joint_names_expr=["gripper_[lr]_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=0.0,
        ),

        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint[1-2]"],
            effort_limit_sim=30.0,
            velocity_limit_sim=2.0,
            stiffness=150.0,
            damping=3.0,
        ),
    },

    soft_joint_pos_limit_factor=1.0,
)


# -----------------------------------------------------------------------------
# 테이블: 외부 USD 직접 로딩
# -----------------------------------------------------------------------------
# Collision 및 정적/Kinematic 설정은 원본 USD에 준비.
table_cfg = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",

    spawn=sim_utils.UsdFileCfg(
        usd_path=TABLE_USD_PATH,
    ),

    init_state=AssetBaseCfg.InitialStateCfg(
        pos=TABLE_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


# -----------------------------------------------------------------------------
# Cube 하나
# -----------------------------------------------------------------------------
cube_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",

    spawn=sim_utils.CuboidCfg(
        size=CUBE_SIZE,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),

        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),

        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.10,
        ),

        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),

        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.5, 0.9),
        ),
    ),

    init_state=RigidObjectCfg.InitialStateCfg(
        pos=CUBE_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


# -----------------------------------------------------------------------------
# 카메라 경로: 원본 USD를 열었을 때의 경로를 작성
# -----------------------------------------------------------------------------
# 아래 경로가 Camera 자체여도 되고, Camera 하나가 들어 있는 부모여도 됨.
RIGHT_WRIST_CAMERA_SOURCE = (
    "/Root/ffw_bg2_follower/right_arm/"
    "arm_r_link7/visuals/d405"
)

# 전달된 두 경로가 같으므로 왼쪽 경로는 아직 지정하지 않음.
# 실제 왼쪽 카메라 경로를 확인하면 None 대신 문자열로 작성.
LEFT_WRIST_CAMERA_SOURCE = None

HEAD_CAMERA_SOURCE = (
    "/Root/ffw_bg2_follower/head/head_link2"
)

# 출력 해상도와 갱신 주기. 사용할 센서 규격에 맞춰 수정.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_UPDATE_PERIOD = 1.0 / 30.0


def _scene_path(stage, source_path):
    """원본 USD 경로를 장면에 불러온 로봇 아래 경로로 변환."""

    default_prim = stage.GetDefaultPrim()

    if not default_prim.IsValid():
        raise ValueError("Robot USD has no defaultPrim.")

    default_path = default_prim.GetPath()

    if not source_path.HasPrefix(default_path):
        raise ValueError(
            f"{source_path} is outside defaultPrim {default_path}"
        )

    relative_path = str(
        source_path.MakeRelativePath(default_path)
    )

    root = "{ENV_REGEX_NS}/Robot"

    if relative_path == ".":
        return root

    return root + "/" + relative_path


def _find_cameras(stage, source_path):
    """지정한 prim 또는 그 아래에 있는 실제 USD Camera 검색."""

    parent = stage.GetPrimAtPath(source_path)

    if not parent.IsValid():
        raise ValueError(
            f"Camera mounting path not found: {source_path}"
        )

    return [
        prim
        for prim in Usd.PrimRange(
            parent,
            Usd.TraverseInstanceProxies(),
        )
        if prim.IsA(UsdGeom.Camera)
    ]


def _existing_camera_cfg(stage, source_path):
    """기존 USD Camera에 연결. 새 카메라를 생성하지 않음."""

    cameras = _find_cameras(stage, source_path)

    if len(cameras) != 1:
        found = [str(prim.GetPath()) for prim in cameras]

        raise ValueError(
            f"Expected one USD Camera under {source_path}, "
            f"found {len(cameras)}: {found}. "
            "Set the source path to one actual Camera prim. "
            "A camera housing mesh alone is not a Camera sensor."
        )

    camera_path = cameras[0].GetPath()
    prim_path = _scene_path(stage, camera_path)

    print(f"[CAMERA] source: {camera_path}")
    print(f"[CAMERA] scene:  {prim_path}")

    return CameraCfg(
        prim_path=prim_path,

        # USD에 이미 존재하는 Camera 사용.
        spawn=None,

        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        update_period=CAMERA_UPDATE_PERIOD,

        data_types=[
            "rgb",
            "distance_to_image_plane",
        ],

        # 손목/머리가 움직일 때 데이터의 카메라 자세도 갱신.
        update_latest_camera_pose=True,
    )


def configure_robot_cameras(scene_cfg, enable_head=False):
    """카메라 설정을 SceneCfg에 추가하고 활성화된 이름을 반환."""

    stage = Usd.Stage.Open(ROBOT_USD_PATH)

    if stage is None:
        raise RuntimeError(
            f"Cannot open robot USD: {ROBOT_USD_PATH}"
        )

    active_names = []

    # 기존 손목 카메라 연결.
    for name, source_path in (
        ("right_wrist_camera", RIGHT_WRIST_CAMERA_SOURCE),
        ("left_wrist_camera", LEFT_WRIST_CAMERA_SOURCE),
    ):
        if source_path is None:
            continue

        setattr(
            scene_cfg,
            name,
            _existing_camera_cfg(stage, source_path),
        )
        active_names.append(name)

    # 머리 카메라는 --head_camera 사용 시에만 활성화.
    if enable_head:
        head_cameras = _find_cameras(stage, HEAD_CAMERA_SOURCE)

        if head_cameras:
            # 나중에 USD에 Camera를 추가했다면 기존 Camera에 연결.
            scene_cfg.head_camera = _existing_camera_cfg(
                stage,
                HEAD_CAMERA_SOURCE,
            )

        else:
            # 아직 Camera가 없다면 head_link2 아래에 새로 생성.
            head_prim = stage.GetPrimAtPath(HEAD_CAMERA_SOURCE)
            parent_path = _scene_path(
                stage,
                head_prim.GetPath(),
            )

            scene_cfg.head_camera = CameraCfg(
                prim_path=parent_path + "/SimHeadCamera",

                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                update_period=CAMERA_UPDATE_PERIOD,

                data_types=[
                    "rgb",
                    "distance_to_image_plane",
                ],

                # 아래 렌즈와 장착 자세는 임시값.
                # 실제 센서의 내참수와 장착 위치/방향에 맞춰 수정.
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.05, 10.0),
                ),

                offset=CameraCfg.OffsetCfg(
                    pos=(0.05, 0.0, 0.05),
                    rot=(1.0, 0.0, 0.0, 0.0),

                    # 부모 링크 기준 +X 방향을 바라보는 설정.
                    convention="world",
                ),

                update_latest_camera_pose=True,
            )

        active_names.append("head_camera")

    return active_names