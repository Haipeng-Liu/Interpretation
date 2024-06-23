#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: model_test_task.py
@Author: haipeng.liu
@Date: 2024/1/6
"""
import os.path
import shutil
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from matplotlib import animation
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.utils.sim_agents import visualizations, submission_specs

from common import TaskType, LoadConfigResultDate
from net_works import BackBone
from tasks import BaseTask
from utils import DataUtil, MathUtil, MapUtil
def average_displacement_error(predicted_trajs, gt_traj, squeeze=True):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    if squeeze:
        return np.min(ade)
    return np.min(ade, axis=0)


def final_displacement_error(predicted_trajs, gt_traj, squeeze=True):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    if squeeze:
        return np.min(final_error)
    return np.min(final_error, axis=0)

class Tester(object):
    def __init__(self, device, model_path):
        self.device = device
        self.load_model(model_path)

    def load_model(self, file_path):
        result_info = LoadConfigResultDate()
        betas = MathUtil.generate_linear_schedule(result_info.train_model_config.time_steps)
        self.model = BackBone(betas).eval()
        device = torch.device("cpu")
        pretrained_dict = torch.load(file_path, map_location=device)
        model_dict = self.model.state_dict()
        # 模型参数赋值
        new_model_dict = dict()
        for key in model_dict.keys():
            if ("module." + key) in pretrained_dict:
                new_model_dict[key] = pretrained_dict["module." + key]
            elif key in pretrained_dict:
                new_model_dict[key] = pretrained_dict[key]
            else:
                print("key: ", key, ", not in pretrained")
        self.model.load_state_dict(new_model_dict)
        print("load_pretrain_model success")

    
    def load_data(self, dataset_path, get_data):
        """Load the testing data from the specified path."""
        match_filenames = tf.io.matching_files(dataset_path)
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data").take(100)
        dataset_iterator = dataset.as_numpy_iterator()
        return get_data(data, self.hyper_parameters["data_scale"], self.device)

    def predict(self, input, mask, initial_pos, output, scenario_bytes):
        scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
        data_dict = DataUtil.transform_data_to_input(scenario, result_info)
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(torch.float32).unsqueeze(dim=0)
        predict_traj = self.model(data_dict)[-1]
        predicted_traj_mask = data_dict['predicted_traj_mask'][0]
        predicted_future_traj = data_dict['predicted_future_traj'][0]
        predicted_his_traj = data_dict['predicted_his_traj'][0]
        predicted_num = 0
        for i in range(predicted_traj_mask.shape[0]):
            if int(predicted_traj_mask[i]) == 1:
                predicted_num += 1
        generate_traj = predict_traj[:predicted_num]
        predicted_future_traj = predicted_future_traj[:predicted_num]
        predicted_his_traj = predicted_his_traj[:predicted_num]
        real_traj = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, :2].detach().numpy()
        real_yaw = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, 2].detach().numpy()
        model_output = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, :2].detach().numpy()
        model_yaw = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, 2].detach().numpy()

        min_ade = (
            average_displacement_error(predicted_futures, output, squeeze=False)
            / self.hyper_parameters["data_scale"]
        )
        min_fde = (
            final_displacement_error(
                predicted_futures, output.transpose(1, 0, 2), squeeze=False
            )
            / self.hyper_parameters["data_scale"]
        )
        return min_ade, min_fde


    
if __name__ == "__main__":
    # show_result()
    pass
