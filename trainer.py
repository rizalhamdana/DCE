import os
import os.path
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import pickle
import wandb
import random
import string


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def _train(args):
    logdir = args["logdir"]
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(3))
    logfilename = (
        "logdir_new/{}/{}_{}_{}_{}_{}_{}_{}_{}_".format(
            logdir,
            args["prefix"],
            args["seed"],
            args["model_name"],
            args["net_type"],
            args["dataset"],
            args["init_cls"],
            args["loss_type"],
            args["order"],
        )
        + time.strftime("%m-%d-%H:%M:%S", time.localtime())
        + random_string
    )
    os.makedirs(logfilename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    print(logfilename)
    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"], args
    )
    args["class_order"] = data_manager._class_order
    model = factory.get_model(args["model_name"], args)
    log_result = []
    cnn_curve, nme_curve = {"top1": []}, {"top1": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        log_result.append({"cnn": cnn_accy, "nme": nme_accy})
        model.after_task()
        log_info = {}
        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        log_info["cnn"] = cnn_accy["top1"]

        wandb.log(log_info, step=task)
        if args["save_model"]==1:
            torch.save(model, os.path.join(logfilename, "task_{}.pth".format(int(task))))
    with open(os.path.join(logfilename, "log_result.pkl"), "wb") as f:
        pickle.dump(log_result, f)
    with open(os.path.join(args["out_dir"], "log_result.pkl"), "wb") as f:
        pickle.dump(log_result, f)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        elif len(device_type) == 1:
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
