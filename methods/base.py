import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from sklearn.covariance import OAS, LedoitWolf

EPSILON = 1e-8
batch_size = 256


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args["memory_per_class"]
        self._fixed_memory = args["fixed_memory"]
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self._margin_sample_num = args["margin_sample_num"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true), decimals=2
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
            else:
                _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source="train", mode="test", appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True
            )
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(selected_exemplars, exemplar_targets)
            )
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info("Constructing exemplars for new classes...({} per classes)".format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True
            )
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(selected_exemplars, exemplar_targets)
            )
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

    def _get_exemplar_with_class_idxes(self, class_idx):
        ex_d, ex_t = np.array([]), np.array([])
        # class_idx = [i for i in class_idx]
        for i in class_idx:
            mask = np.where(self._targets_memory == i)[0]
            ex_d = (
                np.concatenate((ex_d, copy.deepcopy(self._data_memory[mask])))
                if len(ex_d) != 0
                else copy.deepcopy(self._data_memory[mask])
            )
            ex_t = (
                np.concatenate((ex_t, copy.deepcopy(self._targets_memory[mask])))
                if len(ex_t) != 0
                else copy.deepcopy(self._targets_memory[mask])
            )
        return ex_d, ex_t

    def _extract_vectors(self, loader):
        """
        提取向量的方法，返回 torch 张量格式的 vectors 和 targets。
        """
        self._network.eval()
        vectors = []
        targets = []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _inputs = _inputs.to(self._device)
                _targets = _targets.to(self._device)
                if isinstance(self._network, nn.DataParallel):
                    _vectors = self._network.module.extract_vector(_inputs)
                else:
                    _vectors = self._network.extract_vector(_inputs)
                vectors.append(_vectors.detach().cpu())
                targets.append(_targets.detach().cpu())

        vectors = torch.cat(vectors, dim=0)
        targets = torch.cat(targets, dim=0)
        return vectors, targets

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        """
        计算各类别的均值和协方差矩阵。

        参数:
        - data_manager: 数据管理器，用于获取数据集。
        - check_diff: 是否检查均值差异。
        - oracle: 是否使用 oracle 模式。
        """
        # 初始化或扩展 class_means 和 class_covs
        if hasattr(self, "_class_means") and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.size(0)
            assert ori_classes == self._known_classes, "原有类别数量与已知类别数量不匹配。"
            new_class_means = torch.zeros((self._total_classes, self.feature_dim))
            new_class_means[: self._known_classes] = self._class_means
            self._class_means = new_class_means
            self.new_class_num = np.array([0] * self._total_classes)
            self.new_class_num[: self._known_classes] = self._class_num
            self._class_num = self.new_class_num

            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[: self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = torch.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            self._class_num = np.array([0] * self._total_classes)

        # 检查已知类别的均值差异
        if check_diff:
            for class_idx in range(self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True
                )
                idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = vectors.mean(dim=0)
                class_cov = torch.cov(vectors.t())

                # 计算与现有均值的余弦相似度
                existing_mean = self._class_means[class_idx, :].unsqueeze(0)
                similarity = torch.cosine_similarity(existing_mean, class_mean.unsqueeze(0)).item()
                log_info = f"cls {class_idx} sim: {similarity}"
                logging.info(log_info)

                # 保存均值
                torch.save(class_mean, f"task_{self._cur_task}_cls_{class_idx}_mean.pt")

        # Oracle 模式下更新已知类别的均值和协方差
        if oracle:
            for class_idx in range(self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True
                )
                idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                class_mean = vectors.mean(dim=0)
                class_cov = torch.cov(vectors.t()) + torch.eye(self.feature_dim) * 1e-5

                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov

        # 批量处理新类别
        new_classes = np.arange(self._known_classes, self._total_classes)
        data, targets, idx_dataset = data_manager.get_dataset(new_classes, source="train", mode="test", ret_data=True)
        idx_loader = DataLoader(idx_dataset, batch_size=128, shuffle=False, num_workers=4)
        vectors, targets = self._extract_vectors(idx_loader)  # vectors: torch.Tensor, targets: torch.Tensor

        # 遍历新类别并计算均值和协方差
        for class_idx in new_classes:
            class_vectors = vectors[targets == class_idx]
            class_vectors = class_vectors.to(self._device)
            self._class_num[class_idx] = class_vectors.size(0)
            if class_vectors.size(0) < 1:
                logging.warning(f"No vectors found for class {class_idx}. Skipping.")
                continue

            class_mean = class_vectors.mean(dim=0).detach().cpu()
            self._class_means[class_idx, :] = class_mean
            if class_vectors.size(0) >= self._margin_sample_num:
                oas = OAS()
                class_cov = (oas.fit(tensor2numpy(class_vectors))).covariance_
                class_cov = torch.from_numpy(class_cov).float()
                # 更新类别均值和协方差
                self._class_covs[class_idx, ...] = class_cov

        # 后处理：对于样本数量少于20的类别，使用其他类别的平均协方差矩阵
        # 仅在 self._known_classes 和 self._total_classes 范围内
        relevant_classes = np.arange(self._known_classes, self._total_classes)
        relevant_class_num = self._class_num[relevant_classes]
        sufficient_samples_mask = relevant_class_num >= self._margin_sample_num

        if sufficient_samples_mask.sum() > 0:
            # 获取样本数量大于等于20的类别的索引
            valid_classes = relevant_classes[sufficient_samples_mask]
            # 计算这些类别的平均协方差
            average_cov = self._class_covs[valid_classes].mean(dim=0)
        else:
            # 如果没有任何类别满足样本数量要求，使用单位矩阵作为默认协方差
            average_cov = torch.eye(self.feature_dim)
            logging.warning(
                "在 known_classes 和 total_classes 范围内，没有任何类别的样本数量达到20个，使用单位矩阵作为默认协方差。"
            )

        # 替换样本数量少于20的类别的协方差矩阵
        for class_idx in relevant_classes:
            self._class_covs[class_idx, ...] = average_cov
                # logging.info(f"Class {class_idx} has {self._class_num[class_idx]} samples. 使用平均协方差矩阵。")

        # 如果需要返回某些值，可以在此处添加
        return
