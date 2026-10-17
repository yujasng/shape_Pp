"""FFW-BG2 양손 그리퍼 개별 개폐. right: F9(close) F10(open) left : F11(open) F12(close)"""
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np


SCENE = Path(__file__).resolve().with_name("scene_shape.xml")
GRIPPERS = {
    "right": {"prefix": "gripper_r_", "open_ratio": 0.0, "close_ratio": 1.0},
    "left": {"prefix": "gripper_l_", "open_ratio": 0.0, "close_ratio": 1.0},
}
MOVE_TIME = 1.0                  


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    hold = data.ctrl.copy()
    grip, lower, upper, hands = [], [], [], []

    for aid in range(model.nu):
        # joint position/PD actuator
        if (model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_AFFINE):
            raise RuntimeError(
                f"position actuator 설정을 확인해주세요: {model.actuator(aid).name}"
            )

        jid = int(model.actuator_trnid[aid, 0])
        gain = float(model.actuator_gainprm[aid, 0])
        b0, b1 = model.actuator_biasprm[aid, :2]
        gear = float(model.actuator_gear[aid, 0])
        if (gain <= 0 or b1 >= 0 or abs(gear) < 1e-12
                or model.jnt_type[jid] not in
                (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)):
            raise RuntimeError(
                f"지원하지 않는 actuator 설정: {model.actuator(aid).name}"
            )

        # other joints stay
        hold[aid] = -(b0 + b1 * data.actuator_length[aid]) / gain
        if model.actuator_ctrllimited[aid]:
            hold[aid] = np.clip(hold[aid], *model.actuator_ctrlrange[aid])

        #joint 
        joint_name = model.joint(jid).name
        hand = next((side for side, cfg in GRIPPERS.items()
                     if joint_name.startswith(cfg["prefix"])), None)
        if hand is None:
            continue
        if not model.actuator_ctrllimited[aid]:
            raise RuntimeError("그리퍼 position actuator의 ctrlrange를 지정해주세요.")

        lo, hi = model.actuator_ctrlrange[aid].copy()
        if model.jnt_limited[jid]:
            allowed = -(b0 + b1 * gear * model.jnt_range[jid]) / gain
            lo, hi = max(lo, allowed.min()), min(hi, allowed.max())
        if hi <= lo:
            raise RuntimeError("그리퍼의 joint range와 ctrlrange를 확인해주세요.")

        grip.append(aid)
        lower.append(lo)
        upper.append(hi)
        hands.append(hand)

    missing = [side for side in GRIPPERS if side not in hands]
    if missing:
        names = [model.joint(j).name for j in range(model.njnt)]
        raise RuntimeError(
            f"그리퍼 actuator를 찾지 못했습니다: {missing}. joints={names}"
        )

    lower, upper = np.array(lower), np.array(upper)
    masks = {side: np.array([hand == side for hand in hands]) for side in GRIPPERS}
    open_ratio = np.array([GRIPPERS[hand]["open_ratio"] for hand in hands])
    close_ratio = np.array([GRIPPERS[hand]["close_ratio"] for hand in hands])
    opened = lower + open_ratio * (upper - lower)
    closed = lower + close_ratio * (upper - lower)
    command = np.clip(hold[grip], lower, upper)
    target = command.copy()
    dt = float(model.opt.timestep)
    max_change = (upper - lower) * dt / MOVE_TIME

    for hand, aid, u_open, u_close in zip(hands, grip, opened, closed):
        print(f"[{hand}] {model.actuator(aid).name}: open={u_open:.4f}, close={u_close:.4f}")
    print("viewer 창을 클릭한 뒤 오른손 F9: 열기 / F10: 닫기")
    print("왼손 F11: 열기 / F12: 닫기")
    actions = {
        glfw.KEY_F9: ("right", "open"), glfw.KEY_F10: ("right", "close"),
        glfw.KEY_F11: ("left", "open"), glfw.KEY_F12: ("left", "close"),
    }

    keys = SimpleQueue()
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        wrists = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                  for name in ("arm_r_link7", "arm_l_link7")]
        wrists = [body for body in wrists if body >= 0]
        if wrists:
            with viewer.lock():
                viewer.cam.lookat[:] = data.xpos[wrists].mean(axis=0)
                viewer.cam.distance = 1.4
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20

        while viewer.is_running():
            start = time.perf_counter()
            with viewer.lock():
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    if key in actions:
                        hand, action = actions[key]
                        mask = masks[hand]
                        values = opened if action == "open" else closed
                        target[mask] = values[mask]
                        print(f"[{hand.upper()} {action.upper()}]")

                # 목표값까지 서서히 이동.
                command += np.clip(target - command, -max_change, max_change)
                data.ctrl[:] = hold
                data.ctrl[grip] = command
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
