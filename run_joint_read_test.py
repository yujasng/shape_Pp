from pathlib import Path
import math
import time

import mujoco
import mujoco.viewer

SCENE_PATH = Path(__file__).resolve().with_name("scene_shape.xml")

PRINT_INTERVAL = 0.5  


def print_joint_positions(model, data, joint_ids):
    lines = [
        f"\n----- sim_time={data.time:.3f} s -----",
        "[JOINT POSITIONS]",
    ]

    for jid in joint_ids:
        address = int(model.jnt_qposadr[jid])
        value = float(data.qpos[address])
        name = model.joint(jid).name or f"joint_{jid}"

        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            text = (
                f"{value:+.5f} rad "
                f"({math.degrees(value):+.2f} deg)"
            )
        else:
            text = f"{value:+.5f} m"

        lines.append(f"  {name:28s} {text}")

    print("\n".join(lines), flush=True)


def main():
    if not math.isfinite(PRINT_INTERVAL) or PRINT_INTERVAL <= 0:
        raise ValueError("PRINT_INTERVAL은 양수여야 합니다.")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    home = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)

    mujoco.mj_forward(model, data)

    # 회전/직선 관절 선택. 수동 관절도 포함한다.
    scalar = (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE,
    )
    joint_ids = [
        jid for jid in range(model.njnt)
        if model.jnt_type[jid] in scalar
    ]

    next_print = 0.0
    dt = float(model.opt.timestep)

    # MuJoCo 실행.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()

            with viewer.lock():

                mujoco.mj_step(model, data)

                # 관절 위치 출력.
                now = time.perf_counter()
                if now >= next_print:
                    print_joint_positions(model, data, joint_ids)
                    next_print = now + PRINT_INTERVAL

            viewer.sync()

            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass