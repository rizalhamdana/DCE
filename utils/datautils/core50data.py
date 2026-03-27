#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 23-07-2019                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Data Loader for the CORe50 Dataset """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# other imports
import numpy as np
import pickle as pkl
import os
import logging
from hashlib import md5
from PIL import Image
import random


class CORE50(object):
    """CORe50 Data Loader calss
    Args:
        root (string): Root directory of the dataset where ``core50_128x128``,
            ``paths.pkl``, ``LUP.pkl``, ``labels.pkl``, ``core50_imgs.npz``
            live. For example ``~/data/core50``.
        preload (string, optional): If True data is pre-loaded with look-up
            tables. RAM usage may be high.
        scenario (string, optional): One of the three scenarios of the CORe50
            benchmark ``ni``, ``nc``, ``nic``, `nicv2_79`,``nicv2_196`` and
             ``nicv2_391``.
        train (bool, optional): If True, creates the dataset from the training
            set, otherwise creates from test set.
        cumul (bool, optional): If True the cumulative scenario is assumed, the
            incremental scenario otherwise. Practically speaking ``cumul=True``
            means that for batch=i also batch=0,...i-1 will be added to the
            available training data.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed as in the official benchmark.
        start_batch (int, optional): One of the training incremental batches
            from 0 to max-batch - 1. Remember that for the ``ni``, ``nc`` and
            ``nic`` we have respectively 8, 9 and 79 incremental batches. If
            ``train=False`` this parameter will be ignored.
    """

    nbatch = {"ni": 8, "nc": 9, "nic": 79, "nicv2_79": 79, "nicv2_196": 196, "nicv2_391": 391}

    def __init__(self, root="", preload=False, scenario="ni", cumul=False, run=0, start_batch=0, order=0):
        """ " Initialize Object"""

        self.root = os.path.expanduser(root)
        self.preload = preload
        self.scenario = scenario
        self.cumul = cumul
        self.run = run
        self.batch = start_batch

        if preload:
            print("Loading data...")
            bin_path = os.path.join(root, "core50_imgs.bin")
            if os.path.exists(bin_path):
                with open(bin_path, "rb") as f:
                    self.x = np.fromfile(f, dtype=np.uint8).reshape(164866, 128, 128, 3)

            else:
                with open(os.path.join(root, "core50_imgs.npz"), "rb") as f:
                    npzfile = np.load(f)
                    self.x = npzfile["x"]
                    print("Writing bin for fast reloading...")
                    self.x.tofile(bin_path)

        print("Loading paths...")
        with open(os.path.join(root, "paths.pkl"), "rb") as f:
            self.paths = pkl.load(f)

        print("Loading LUP...")
        with open(os.path.join(root, "LUP.pkl"), "rb") as f:
            self.LUP = pkl.load(f)

        print("Loading labels...")
        with open(os.path.join(root, "labels.pkl"), "rb") as f:
            self.labels = pkl.load(f)

        data_file_path = os.path.join("utils/datautils", "core50.pkl")
        if os.path.exists(data_file_path):
            print("Loading data info...")
            with open(data_file_path, "rb") as f:
                self.data_info_list = pkl.load(f)
        # self.split_dataset()
        self.data_info_list = self.order_data(self.data_info_list, order)

    def __iter__(self):
        return self
    
    def order_data(self,data_info_list,order):
        
        order_1 = [8, 3, 2, 7, 1, 5, 4, 6]
        order_2 = [2, 7, 1, 5, 4, 6, 8, 3]
        order_3 = [3, 1, 7, 2, 4, 5, 6, 8]
        order_4 = [1, 7, 2, 4, 5, 6, 8, 3]
        order_5 = [7, 2, 4, 5, 6, 8, 3, 1]
        order_list = [order_1, order_2, order_3, order_4, order_5]
        order = order_list[order-1]
        data_info_list_new = []
        for i in range(len(order)):
            data_info_list_new.append(data_info_list[order[i]-1])
        return data_info_list_new
    
    

    def get_data_batchidx(self, idx):

        scen = self.scenario
        run = self.run
        batch = idx

        if self.batch == self.nbatch[scen]:
            raise StopIteration

        # Getting the right indexis
        if self.cumul:
            train_idx_list = []
            for i in range(self.batch + 1):
                train_idx_list += self.LUP[scen][run][i]
        else:
            train_idx_list = self.LUP[scen][run][batch]

        # loading data
        if self.preload:
            train_x = np.take(self.x, train_idx_list, axis=0).astype(np.float32)
        else:
            print("Loading data...")
            # Getting the actual paths
            train_paths = []
            for idx in train_idx_list:
                train_paths.append(os.path.join(self.root, self.paths[idx]))
            # loading imgs
            train_x = self.get_batch_from_paths(train_paths).astype(np.float32)

        # In either case we have already loaded the y
        if self.cumul:
            train_y = []
            for i in range(self.batch + 1):
                train_y += self.labels[scen][run][i]
        else:
            train_y = self.labels[scen][run][batch]

        train_y = np.asarray(train_y, dtype=np.int)

        return (train_x, train_y)


    def split_dataset(self):
        scen = self.scenario
        run = self.run
        batch = self.batch
        data_info_list = []
        domain_num = self.nbatch[scen]
        imb_factor = [100, 100, 100, 100, 50, 50, 50, 50]
        for domain in range(domain_num):
            domain_x = self.LUP[scen][run][domain]
            domain_y = self.labels[scen][run][domain]
            domain_cls_num = len(set(domain_y))
            self.count_cls_data(domain_y)
            data_info = self.make_imb_dataset(
                np.array(domain_x), np.array(domain_y), imb_factor[domain]
            )
            data_info_list.append(data_info)
        with open(os.path.join(self.root, "datainfo.pkl"), "wb") as f:
            pkl.dump(data_info_list, f)
        return data_info_list

    def count_cls_data(self, domain_y):
        domain_cls_num = len(set(domain_y))
        domain_cls_count = []
        for i in range(domain_cls_num):
            domain_cls_count_i = np.sum(np.array(domain_y) == i)
            domain_cls_count.append(domain_cls_count_i)
        print("Cls num:{}".format(domain_cls_num))
        print("Domain cls count:{}".format(domain_cls_count))

    def make_imb_dataset(self, domain_x, domain_y, imb_factor):
        img_num_list = self.gen_cls_data_num(50, imb_factor)
        random.shuffle(img_num_list)
        for i in range(50):
            cls_idx = i
            cls_num = img_num_list[cls_idx]
            cls_idx_list = np.where(domain_y == cls_idx)
            domain_x = np.delete(domain_x, cls_idx_list[0][cls_num:], axis=0)
            domain_y = np.delete(domain_y, cls_idx_list[0][cls_num:], axis=0)
        return {"x": domain_x, "y": domain_y, "cls_num": img_num_list}

    def gen_cls_data_num(self, cls_num, imb_factor):
        img_max = 300
        imb_factor = 1 / imb_factor
        img_num_list = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_list.append(int(num))
        return img_num_list

    def get_test_set(self):
        """Return the test set (the same for each inc. batch)."""

        scen = self.scenario
        run = self.run

        test_idx_list = self.LUP[scen][run][-1]

        if self.preload:
            test_x = np.take(self.x, test_idx_list, axis=0).astype(np.float32)
        else:
            # test paths
            test_paths = []
            for idx in test_idx_list:
                test_paths.append(os.path.join(self.root, self.paths[idx]))

            # test imgs
            test_x = self.get_batch_from_paths(test_paths).astype(np.float32)

        test_y = self.labels[scen][run][-1]
        test_y = np.asarray(test_y, dtype=np.int32)

        return test_x, test_y

    # next = __next__  # python2.x compatibility.

    @staticmethod
    def get_batch_from_paths(paths, compress=False, snap_dir="", on_the_fly=True, verbose=False):
        """Given a number of abs. paths it returns the numpy array
        of all the images."""

        # Getting root logger
        log = logging.getLogger("mylogger")

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = len(paths)
        hexdigest = md5("".join(paths).encode("utf-8")).hexdigest()
        log.debug("Paths Hex: " + str(hexdigest))
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, "rb") as f:
                    npzfile = np.load(f)
                    x, y = npzfile["x"]
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, "rb") as f:
                    x = np.fromfile(f, dtype=np.uint8).reshape(num_imgs, 128, 128, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end="")
                x[i] = np.array(Image.open(path))

            if verbose:
                print()

            if not on_the_fly:
                # Then we save x
                if compress:
                    with open(file_path, "wb") as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert x is not None, "Problems loading data. x is None!"

        return x
