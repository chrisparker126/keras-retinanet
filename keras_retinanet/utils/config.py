"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import configparser
import numpy as np
import keras
from ..utils.anchors import AnchorParameters
import csv

def read_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    return config


def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)

def read_transformation_kw_args(transform_file_path):
    with open(transform_file_path, 'r') as file:
        return parse_transform_file(file)

def parse_transform_file(file):
    csv_file = csv.writer(file, delimiter=',')
    kw_args = {}
    for row in csv_file:
        if row[1] = 'i':
            kw_args[row[0]] = int(row[2]
        elif row[1] = 't':
            kw_args[row[0]] = (int(row[2]), int(row[3]))
    return kw_args


