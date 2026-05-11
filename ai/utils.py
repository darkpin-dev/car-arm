import argparse
import importlib
import os
import sys


def train_setting():
    parser = argparse.ArgumentParser(description="CarArm RL Training")
    parser.add_argument("--env", type=str, required=True, help="환경 파일 경로")
    parser.add_argument("--name", type=str, required=True, help="이름")
    parser.add_argument("--device", type=str, required=True, help="장치 이름, 번호")
    parser.add_argument("--step", type=int, required=True, help="학습 스텝 수")

    args = parser.parse_args()

    module_name = os.path.splitext(os.path.basename(args.env))[0]
    spec = importlib.util.spec_from_file_location(module_name, args.env)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    args.env = module.CarArmEnv
    args.logs_dir = f"./logs/{args.name}"
    args.models_dir = f"./models/{args.name}"

    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    return args

def test_setting():
    parser = argparse.ArgumentParser(description="CarArm RL Training")
    parser.add_argument("--env", type=str, required=True, help="환경 파일 경로")
    parser.add_argument("--name", type=str, required=True, help="이름")
    parser.add_argument("--device", type=str, required=True, help="장치 이름, 번호")

    args = parser.parse_args()

    module_name = os.path.splitext(os.path.basename(args.env))[0]
    spec = importlib.util.spec_from_file_location(module_name, args.env)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    args.env = module.CarArmEnv
    args.models_dir = f"./models/{args.name}"

    return args
