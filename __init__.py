"""환경 패키지 및 실행 진입점."""


def main():
    import argparse
    import importlib
    from pathlib import Path
    import sys

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="BG2 + table + cube + robot cameras"
    )

    # 환경 step 수. 0이면 창을 닫을 때까지 실행.
    parser.add_argument("--steps", type=int, default=0)

    # 머리 카메라 활성화. --enable_cameras와 함께 사용.
    parser.add_argument("--head_camera", action="store_true")

    # --headless, --device, --enable_cameras 등 실행 옵션 추가.
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be 0 or a positive integer")

    if args.head_camera and not args.enable_cameras:
        parser.error("--head_camera requires --enable_cameras")

    # Isaac Sim을 먼저 실행.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env = None

    try:
        import torch
        from isaaclab.envs import ManagerBasedEnv

        package_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(package_dir.parent))

        cfg_module = importlib.import_module(
            f"{package_dir.name}.env_cfg"
        )
        objects_module = importlib.import_module(
            f"{package_dir.name}.scene_objects_cfg"
        )

        env_cfg = cfg_module.ShapeEnvCfg()
        env_cfg.sim.device = args.device

        # 활성화한 카메라의 scene 이름 목록.
        camera_names = []

        if args.enable_cameras:
            camera_names = objects_module.configure_robot_cameras(
                env_cfg.scene,
                enable_head=args.head_camera,
            )

        env = ManagerBasedEnv(cfg=env_cfg)

        with torch.inference_mode():
            env.reset()

            # 초기 자세를 유지하는 action.
            actions = torch.zeros_like(env.action_manager.action)

            print("[ROBOT] joints:", env.scene["robot"].joint_names)
            print("[CAMERA] active sensors:", camera_names)

            count = 0

            while simulation_app.is_running():
                if args.steps > 0 and count >= args.steps:
                    break

                if not env.sim.is_playing():
                    env.sim.render()
                    continue

                # 추후 ROS 또는 policy 명령을 actions에 연결.
                observations, info = env.step(actions)
                count += 1

                # 매 환경 step에서 접근하여 센서 주기에 맞춰 갱신.
                camera_outputs = {
                    name: env.scene[name].data.output
                    for name in camera_names
                }

                # 기본 설정에서 시뮬레이션 시간 2초마다 출력.
                if count % 120 == 0:
                    gt = cfg_module.get_ground_truth(env)

                    print(
                        "[GT] cube world xyz:",
                        gt["cube"]["position_w"][0].cpu().tolist(),
                    )

                    for name, output in camera_outputs.items():
                        print(
                            f"[{name}] RGB:",
                            tuple(output["rgb"].shape),
                            "depth:",
                            tuple(
                                output["distance_to_image_plane"].shape
                            ),
                        )

    finally:
        try:
            if env is not None:
                env.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
else:
    from .env_cfg import ShapeEnvCfg

    __all__ = ["ShapeEnvCfg"]