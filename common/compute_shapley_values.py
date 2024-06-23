# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os

import torch
import dill
import random
import numpy as np

from common import shapley_values
from tasks import model_test_task as test


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_file", type=str)      # models/pecnet/sport.pt
    parser.add_argument("dataset_path", type=str)   # data/sport/pecnet_test.pkl
    parser.add_argument("scene_index", help="scene index", type=int)    # 0
    parser.add_argument("output_path", help="result directory", type=str)   # results/sport/pecnet
    return parser.parse_args()

def initialize_device_and_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return device

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parameters = parse_arguments()
    device = initialize_device_and_seed(0)
    tester = test.Tester(device, parameters.load_file)

    shapley_values_estimator = shapley_values.ShapleyValues(tester)
    output_file_path = parameters.output_path + "_agent_%d_ade.pkl" % (parameters.scene_index)

    if os.path.exists(output_file_path):
        raise FileExistsError(f"Output file already exists {output_file_path}.")
    
    result = shapley_values_estimator.run(parameters.dataset_path, parameters.scene_index)

    with open(output_file_path, "wb") as file_writer:
        dill.dump(result, file_writer)
