"""관절을 하나씩 조금 움직였다가 초기 목표로 복귀."""
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


SCENE_PATH = Path(__file__).resolve().with_name("scene_shape.xml")

ANGLE_STEP = np.deg2rad(30.0)  # 회전 관절 이동량 [rad]
LINEAR_STEP = 0.01           # 직선 관절 이동량 [m]
MOVE_TIME = 2.0              # 한 관절의 왕복 시간 [s]
REST_TIME = 0.5              # 복귀 후 대기 시간 [s]
START_DELAY = 1.0            # 시작 전 대기 시간 [s]
PRINT_INTERVAL = 0.05         # 실제 시간 기준 출력 간격 [s]

# None: 독립 구동 가능한 모든 position 관절 테스트.
# 특정 관절만 테스트하려면 아래처럼 설정.
# JOINT_NAMES = ["arm_r_joint1", "head_joint1"]
JOINT_NAMES = None


def set_joint_control(ctrl, group, q):
    # 같은 관절에 actuator가 여러 개 있어도 같은 관절 목표 전달.
    for aid, scale, bias in group["servos"]:
        ctrl[aid] = scale * q + bias


def read_servos(model, data):
    """position/PD actuator의 입력과 관절 위치 사이 관계 계산."""
    groups = {}

    scalar = (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE,
    )

    for aid in range(model.nu):
        supported = (
            model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_dyntype[aid] == mujoco.mjtDyn.mjDYN_NONE
            and model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
            and model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE
        )

        if supported:
            jid = int(model.actuator_trnid[aid, 0])
            gain = float(model.actuator_gainprm[aid, 0])
            b0, b1 = model.actuator_biasprm[aid, :2]
            gear = float(model.actuator_gear[aid, 0])

            supported = (
                model.jnt_type[jid] in scalar
                and gain > 0
                and b1 < 0
                and abs(gear) > 1e-12
            )

        if not supported:
            print(
                "[SKIP] position/PD 방식이 아닌 actuator: "
                f"{model.actuator(aid).name}"
            )
            continue

        qadr = int(model.jnt_qposadr[jid])
        q = float(data.qpos[qadr])

        length_offset = float(data.actuator_length[aid]) - gear * q

        # actuator 입력 u = scale * 관절 위치 q + bias.
        scale = -b1 * gear / gain
        bias = -(b0 + b1 * length_offset) / gain

        low, high = -np.inf, np.inf

        if model.jnt_limited[jid]:
            low, high = model.jnt_range[jid]

        if model.actuator_ctrllimited[aid]:
            bounds = (model.actuator_ctrlrange[aid] - bias) / scale
            low = max(low, float(bounds.min()))
            high = min(high, float(bounds.max()))

        if jid not in groups:
            groups[jid] = {
                "name": model.joint(jid).name or f"joint_{jid}",
                "qadr": qadr,
                "kind": model.jnt_type[jid],
                "low": low,
                "high": high,
                "servos": [],
            }

        group = groups[jid]
        group["low"] = max(group["low"], low)
        group["high"] = min(group["high"], high)
        group["servos"].append((aid, scale, bias))

    hold = data.ctrl.copy()

    for group in groups.values():
        if group["low"] > group["high"]:
            raise ValueError(
                f"{group['name']}: 관절/actuator 허용 범위가 "
                "맞지 않습니다."
            )

        group["q0"] = float(
            np.clip(
                data.qpos[group["qadr"]],
                group["low"],
                group["high"],
            )
        )

        set_joint_control(hold, group, group["q0"])

    return groups, hold


def choose_delta(q0, low, high, maximum):
    """관절 제한을 고려해 이동 방향과 크기 결정."""
    positive = max(0.0, high - q0)
    negative = max(0.0, q0 - low)

    if positive >= maximum:
        return maximum

    if negative >= maximum:
        return -maximum

    if positive >= negative:
        return min(maximum, 0.8 * positive)

    return -min(maximum, 0.8 * negative)


def format_position(group, q):
    if group["kind"] == mujoco.mjtJoint.mjJNT_HINGE:
        return f"{q:+.5f} rad ({np.rad2deg(q):+.2f} deg)"

    return f"{q:+.5f} m"


def main():
    for value in (
        ANGLE_STEP,
        LINEAR_STEP,
        MOVE_TIME,
        REST_TIME,
        PRINT_INTERVAL,
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("이동량과 시간 설정은 양수여야 합니다.")

    if not np.isfinite(START_DELAY) or START_DELAY < 0:
        raise ValueError("START_DELAY는 0 이상이어야 합니다.")

    # 장면 생성.
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    home = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)

    mujoco.mj_forward(model, data)

    groups, hold = read_servos(model, data)

    # joint equality에서 joint1은 다른 관절을 따르는 종속 관절.
    followers = {
        int(model.eq_obj1id[eid])
        for eid in range(model.neq)
        if (
            data.eq_active[eid]
            and model.eq_type[eid] == mujoco.mjtEq.mjEQ_JOINT
        )
    }

    plan = []

    for jid in sorted(groups):
        group = groups[jid]

        if jid in followers:
            print(
                f"[LINKED] {group['name']}: "
                "독립 왕복 테스트 생략"
            )
            continue

        if (
            JOINT_NAMES is not None
            and group["name"] not in JOINT_NAMES
        ):
            continue

        maximum = (
            ANGLE_STEP
            if group["kind"] == mujoco.mjtJoint.mjJNT_HINGE
            else LINEAR_STEP
        )

        group["delta"] = choose_delta(
            group["q0"],
            group["low"],
            group["high"],
            maximum,
        )

        if abs(group["delta"]) < 1e-10:
            print(
                f"[SKIP] {group['name']}: "
                "움직일 수 있는 범위가 없습니다."
            )
            continue

        plan.append(group)

    if JOINT_NAMES is not None:
        missing = set(JOINT_NAMES) - {
            group["name"] for group in plan
        }

        if missing:
            raise ValueError(
                "독립 테스트가 불가능하거나 없는 관절 이름: "
                f"{sorted(missing)}"
            )

    if not plan:
        raise ValueError(
            "테스트 가능한 독립 position/PD 관절이 없습니다."
        )

    print(f"\n총 {len(plan)}개 관절을 순서대로 테스트합니다.")

    for i, group in enumerate(plan, 1):
        print(
            f"  {i:2d}. {group['name']} "
            f"| 이동량={format_position(group, group['delta'])}"
        )

    data.ctrl[:] = hold

    start_time = float(data.time)
    cycle_time = MOVE_TIME + REST_TIME
    last_index = -1
    next_print = 0.0
    dt = float(model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()

            with viewer.lock():
                elapsed = (
                    float(data.time) - start_time - START_DELAY
                )

                if elapsed < 0:
                    index = -1
                else:
                    index = min(
                        int(elapsed / cycle_time), len(plan)
                    )

                # 다음 관절로 넘어갈 때 결과와 관절 이름 출력.
                if index != last_index:
                    if 0 <= last_index < len(plan):
                        previous = plan[last_index]
                        actual = float(
                            data.qpos[previous["qadr"]]
                        )

                        print(
                            f"[RETURN] {previous['name']}"
                            f" | 실제={format_position(previous, actual)}"
                            f" | 복귀 목표 오차="
                            f"{format_position(previous, actual - previous['q0'])}"
                        )

                    if 0 <= index < len(plan):
                        current = plan[index]

                        print(
                            f"\n[TEST {index + 1}/{len(plan)}] "
                            f"{current['name']}"
                        )
                        print(
                            "  초기 목표="
                            f"{format_position(current, current['q0'])}"
                        )

                    elif index == len(plan):
                        print(
                            "\n[DONE] 전체 테스트 완료. "
                            "초기 위치를 목표로 유지합니다. "
                            "창을 닫으면 종료합니다."
                        )

                    last_index = index

                # 기본적으로 모든 position 관절에 초기 목표 전달.
                data.ctrl[:] = hold

                # actuator가 달린 종속 관절은 실제 위치를 따라가게 하여
                # 모델의 equality 연동을 방해하지 않도록 한다.
                for jid in followers:
                    if jid in groups:
                        group = groups[jid]

                        q = float(data.qpos[group["qadr"]])
                        q = float(
                            np.clip(
                                q, group["low"], group["high"]
                            )
                        )

                        set_joint_control(data.ctrl, group, q)

                # 현재 테스트하는 관절만 왕복 목표를 적용.
                if 0 <= index < len(plan):
                    current = plan[index]
                    phase = elapsed - index * cycle_time

                    if phase < MOVE_TIME:
                        # 0 → 1 → 0으로 부드럽게 변화.
                        weight = np.sin(
                            np.pi * phase / MOVE_TIME
                        ) ** 2
                    else:
                        weight = 0.0

                    command = (
                        current["q0"]
                        + current["delta"] * weight
                    )

                    set_joint_control(
                        data.ctrl, current, command
                    )

                mujoco.mj_step(model, data)

                # 실제 관절 값 출력.
                now = time.perf_counter()

                if (
                    0 <= index < len(plan)
                    and now >= next_print
                ):
                    actual = float(
                        data.qpos[current["qadr"]]
                    )

                    print(
                        f"  t={data.time:.2f}"
                        f" | 목표={format_position(current, command)}"
                        f" | 실제={format_position(current, actual)}"
                    )

                    next_print = now + PRINT_INTERVAL

            viewer.sync()

            time.sleep(
                max(
                    0.0,
                    dt - (time.perf_counter() - started),
                )
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass