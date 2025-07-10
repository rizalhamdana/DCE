import json
import argparse
from trainer import train
import torch
import os
import time
import random
import wandb


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    param["data_path"] = param["data_path"]
    wandb.init(project=param["project_name"], config=param)
    param["out_dir"] = wandb.run.dir
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple continual learning algorthms.")
    parser.add_argument("--loss_type", type=str, default="ce", help="Loss type")
    parser.add_argument("--config", type=str,
                        default="configs/domainnet.json", help="Json file of settings.")
    parser.add_argument("--project_name", type=str, default="", help="Project name of wandb")
    parser.add_argument("--order", type=int, default=1, help="Order of the task")
    parser.add_argument("--logdir", type=str, default="logs_default", help="Directory to save logs")
    parser.add_argument("--temp", type=int, default=1)
    parser.add_argument("--bal_epoch", type=int, default=10)
    parser.add_argument("--prompt_type", type=str, default="one", choices=["one", "no", "all"])
    parser.add_argument("--margin_sample_num", type=int, default=10)
    parser.add_argument("--21k", type=int, default=0)
    parser.add_argument("--save_model", type=int, default=0)
    return parser


# domainnet_slip.json

if __name__ == "__main__":
    sleep_time = random.uniform(0, 5)
    time.sleep(sleep_time)
    if torch.cuda.is_available():
        # 获取当前GPU的名称
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Current GPU: {gpu_name}")

        # 检查是否为A100 GPU
        if "A100" in gpu_name:
            # 启用TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            print("TF32 has been enabled for A100 GPU.")
        else:
            print("Current GPU is not A100. TF32 will not be enabled.")
    else:
        print("No GPU available.")
    main()
