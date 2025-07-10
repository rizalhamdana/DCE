import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy_domain, accuracy_domain_shot
from models.DceNet import DceNet
from utils.dataloader import ClassAwareSampler
from torch.distributions.multivariate_normal import MultivariateNormal


class DCE(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self._network = DceNet(args)
        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]

        self.loss_type = args["loss_type"]
        self.cls_split = []
        self.num_each_cls = []
        self.logit_norm = None
        self.bal_epoch = args["bal_epoch"]
        self.use_sacle = 1
        self.prompt_type = args["prompt_type"]
        self.small_lr = 1.0
        self.topk = 2  # origin is 5
        self.class_num = self._network.class_num

        self.all_keys = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        test_dataset_domain = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="test", mode="test"
        )
        self.split_cls(train_dataset.labels)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.test_loader_domain = DataLoader(
            test_dataset_domain,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        sampler = ClassAwareSampler(train_dataset, num_samples_cls=1, is_infinite=True)
        self.resample_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler
        )
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(
            self.train_loader, self.test_loader, self.test_loader_domain, self.resample_loader
        )
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def split_cls(self, targets):
        num_each_cls = []
        for i in range(self.class_num):
            cls_count = ((targets % self.class_num) == i).sum().item()
            if cls_count == 0:
                num_each_cls.append(0.1)
            else:
                num_each_cls.append(cls_count)

        # 对类别样本数量进行排序
        sorted_indices = sorted(
            range(len(num_each_cls)), key=lambda k: num_each_cls[k], reverse=True
        )

        # 计算 head_cls 和 tail_cls 的分界点
        midpoint = len(sorted_indices) // 2

        # 前一半类别作为 head_cls，后一半类别作为 tail_cls
        head_cls = sorted_indices[:midpoint]
        tail_cls = sorted_indices[midpoint:]
        print(num_each_cls)
        self.num_each_cls.append(num_each_cls)
        self.cls_split.append([head_cls, tail_cls])

    def _train(self, train_loader, test_loader, test_loader_domain, bal_train_laoader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if isinstance(self._network, nn.DataParallel):
                if "classifier_pool_naive" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "prompt_pool" in name:
                    if self.prompt_type == "all":
                        param.requires_grad_(True)
                    elif self.prompt_type == "one":
                        if self._cur_task == 0:
                            param.requires_grad_(True)
                if "classifier_pool_bal" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "classifier_pool_rev" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
            else:
                if "classifier_pool_naive" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "prompt_pool" in name:
                    if self.prompt_type == "all":
                        param.requires_grad_(True)
                    elif self.prompt_type == "one":
                        if self._cur_task == 0:
                            param.requires_grad_(True)
                if "classifier_pool_bal" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "classifier_pool_rev" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        prompt_pool_params = []
        other_params = []
        for name, param in self._network.named_parameters():
            if "prompt_pool" in name and param.requires_grad:
                prompt_pool_params.append(param)
            elif param.requires_grad:
                other_params.append(param)
        print(f"Parameters to be updated: {enabled}")
        if self._cur_task == 0:
            optimizer = optim.SGD(
                [
                    {"params": other_params, "lr": self.init_lr},
                    {"params": prompt_pool_params, "lr": self.init_lr * self.small_lr},
                ],
                momentum=0.9,
                weight_decay=self.init_weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.init_epoch
            )
            self.run_epoch = self.init_epoch
            self.train_function(
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                test_loader_domain,
                bal_train_laoader,
            )
            self._compute_class_mean(data_manager=self.data_manager)
            self._stage2_compact_classifier(self.class_num)

        else:
            optimizer = optim.SGD(
                [
                    {"params": other_params, "lr": self.init_lr},
                    {"params": prompt_pool_params, "lr": self.init_lr * self.small_lr},
                ],
                momentum=0.9,
                weight_decay=self.init_weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                test_loader_domain,
                bal_train_laoader,
            )
            self._compute_class_mean(data_manager=self.data_manager)
            self._stage2_compact_classifier(self.class_num)

    def get_ens_loss(self, ens_logits, targets, cls_sample_num):
        if self.loss_type == "ce":
            loss = F.cross_entropy(ens_logits, targets)
        elif self.loss_type == "bce":
            cls_sample_num = torch.tensor(self.num_each_cls[self._cur_task]).to(self._device)
            spc = cls_sample_num.type_as(ens_logits)
            spc = spc.unsqueeze(0).expand(ens_logits.shape[0], -1)
            ens_logits = ens_logits + spc.log()
            loss = F.cross_entropy(ens_logits, targets)
        elif self.loss_type == "drw":
            beta = 0.9999
            per_cls_weights = (1.0 - beta) / (1.0 - (beta**cls_sample_num))
            per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
            loss = F.cross_entropy(ens_logits, targets, weight=per_cls_weights)
        return loss

    def train_function(
        self, train_loader, test_loader, optimizer, scheduler, test_loader_domain, bal_train_laoader
    ):
        prog_bar = tqdm(range(self.run_epoch))
        total_iter_time = 0
        iter_num = 0
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                iter_num += 1
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes
                cls_sample_num = torch.tensor(self.num_each_cls[self._cur_task]).to(self._device)
                start_time = time.time()
                outputs = self._network(inputs, train=True, quick=True)
                logits = outputs["logits"]
                loss_naive = F.cross_entropy(logits, targets)
                spc = cls_sample_num.type_as(logits)
                spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
                bal_logits = outputs["bal_logits"] + spc.log()
                loss_bal = F.cross_entropy(bal_logits, targets)
                rev_logits = outputs["rev_logits"] + 2 * spc.log()
                loss_rev = F.cross_entropy(rev_logits, targets)
                loss = loss_naive + loss_bal + loss_rev
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                end_time = time.time()
                total_iter_time += (end_time - start_time)*1000
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy_domain(self._network, test_loader, domain=1)
                test_domain_acc = self._compute_accuracy_domain(
                    self._network, test_loader_domain, domain=0
                )
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Test_domain_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                    test_domain_acc,
                )
                logging.info(info)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain_shot(
            y_pred.T[0],
            y_true,
            self._known_classes,
            class_num=self.class_num,
            many_shot=self.data_manager.many_shot_classes,
            medium_shot=self.data_manager.medium_shot_classes,
            few_shot=self.data_manager.few_shot_classes,
        )
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        iter_num = 0
        total_iter_time = 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            iter_num += 1
            start_time = time.time()
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[
                1
            ]  # [bs, topk]
            end_time = time.time()
            total_iter_time += (end_time - start_time) * 1000
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return (
            np.concatenate(y_pred),
            np.concatenate(y_true),
        )  # [N, topk]

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        if hasattr(self, "_class_means"):
            nme_accy = None
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _compute_accuracy_domain(self, model, loader, domain=0):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # 0是1阶段 domain，1是1阶段 all，2是2阶段
                if domain == 0:
                    outputs = model(inputs)["last_logits"]
                elif domain == 1:
                    outputs = model(inputs)["all_logits_bal"]
                elif domain == 2:
                    outputs = model(inputs)["logits"]

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return float(np.around(tensor2numpy(correct) * 100 / total, decimals=2))

    def get_weight(self, epoch, epoch_num, weight_type):
        # 如果 weight_type 是 step,前2/3个epoch时0,后面是1
        # 如果 weight_type 是 linear,前2/3个epoch时从0到1,后面是1
        if weight_type == "step":
            if epoch < 2 * epoch_num / 3:
                return 0
            else:
                return 1
        elif weight_type == "linear":
            if epoch < 2 * epoch_num / 3:
                return epoch / (2 * epoch_num / 3)
            else:
                return 1
        elif weight_type == "zero":
            return 0

    def _stage2_compact_classifier(self, task_size):
        for n, p in self._network.select_network.named_parameters():
            p.requires_grad = True
            if "scales" in n:
                p.requires_grad = False
                print("fixed ", n)

        run_epochs = self.bal_epoch
        crct_num = self._total_classes
        param_list = self._network.get_domain_param_list()
        network_params = [
            {"params": param_list, "lr": self.lrate, "weight_decay": self.weight_decay}
        ]
        optimizer = optim.SGD(
            network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        if run_epochs < 5 and self._cur_task == 0:
            run_epochs = run_epochs * 2
        total_iter_time = 0
        iter_num = 0
        for epoch in range(run_epochs):
            losses = 0.0

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
            batch = 0
            for c_id in range(crct_num):
                if self._class_num[c_id] == 0:
                    continue
                batch += 1
                t_id = c_id // task_size
                cls_mean = self._class_means[c_id].detach().to(self._device)
                cls_cov = self._class_covs[c_id].detach().to(self._device)
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                # a more robust implementation of feature scaling
                # fixed scaling during inference, dynamic during training
                if self.use_sacle == 1:
                    rand_scaling = 0.02 * (
                        torch.rand(sampled_data_single.size(0), device=self._device) - 0.5
                    )
                    rand_scaling = 1 / (1 + rand_scaling * (self._cur_task - t_id))
                    sampled_data_single = rand_scaling.unsqueeze(1) * sampled_data_single
                elif self.use_sacle == 2:
                    rand_scaling = 0.02 * (
                        torch.rand(sampled_data_single.size(0), device=self._device) - 0.5
                    )
                    rand_scaling = 1 / (1 + rand_scaling * self._cur_task / 2)
                    sampled_data_single = rand_scaling.unsqueeze(1) * sampled_data_single
                elif self.use_sacle == 3:
                    rand_scaling = 0.02 * (
                        torch.rand(sampled_data_single.size(0), device=self._device) - 0.5
                    )
                    rand_scaling = 1 / (1 + rand_scaling)
                    sampled_data_single = rand_scaling.unsqueeze(1) * sampled_data_single
                elif self.use_sacle == 4:
                    rand_scaling = 0.02 * (
                        torch.rand(sampled_data_single.size(0), device=self._device) - 0.5
                    )
                    rand_scaling = 1 / (1 + rand_scaling * self._cur_task)
                    sampled_data_single = rand_scaling.unsqueeze(1) * sampled_data_single

                sampled_data.append(sampled_data_single.cpu())
                sampled_label.extend([c_id % task_size] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().cpu()
            sampled_label = torch.tensor(sampled_label).long().cpu()

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(batch):
                start_time = time.time()
                iter_num += 1
                inp = inputs[_iter * num_sampled_pcls: (_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls: (_iter + 1) * num_sampled_pcls]
                inp = inp.to(self._device)
                tgt = tgt.to(self._device)
                logits, _, _, _, _ = self._network.forward_head(inp)

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = (
                            torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True)
                            + 1e-7
                        )
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)

                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                end_time = time.time()
                total_iter_time += (end_time - start_time) * 1000

            scheduler.step()
            test_acc = self._compute_accuracy_domain(self._network, self.test_loader, domain=2)
            info = "Stage2 Task {} => Loss {:.3f}, Test_accy {:.3f}".format(
                self._cur_task, losses / self._total_classes, test_acc
            )
            logging.info(info)

    # Debug checks for mean and covariance
    def check_distribution_inputs(self, cls_mean, cls_cov):
        # 1. Check for NaN in inputs
        if torch.isnan(cls_mean).any():
            print("WARNING: NaN in mean vector")
            cls_mean = torch.nan_to_num(cls_mean, nan=0.0)

        if torch.isnan(cls_cov).any():
            print("WARNING: NaN in covariance matrix")
            cls_cov = torch.nan_to_num(cls_cov, nan=1e-6)

        # 2. Ensure covariance is PSD (Positive Semi-Definite)
        min_eig = torch.linalg.eigvals(cls_cov).real.min()
        if min_eig < 0:
            print(f"WARNING: Non-PSD covariance, min eigenvalue: {min_eig}")
            cls_cov = cls_cov + torch.eye(cls_cov.shape[0], device=cls_cov.device) * (
                abs(min_eig) + 1e-6
            )

        return cls_mean, cls_cov
