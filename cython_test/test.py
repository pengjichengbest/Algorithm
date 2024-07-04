# coding:utf-8
"""
    Author: apple
    Date: 2/5/2024
    File: test.py
    ProjectName: Algorithm
    Time: 12:04
"""

import numpy as np
import pandas as pd
from time import time
import difference


def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0], dtype=np.float32)
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean'])
        result[i] = float(groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')].iloc[0])
    return result


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0], dtype=np.float32)
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


if __name__ == '__main__':
    y = np.random.randint(2, size=(5000, 1), dtype=np.int32)
    x = np.random.randint(10, size=(5000, 1), dtype=np.int32)
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    start1 = time()
    result_1 = target_mean_v1(data, 'y', 'x')
    finish1 = time()
    delta1 = finish1 - start1
    start2 = time()
    result_2 = target_mean_v2(data, 'y', 'x')
    finish2 = time()
    delta2 = finish2 - start2
    start3 = time()
    result_3 = difference.get_result(data, 'y', 'x')
    finish3 = time()
    delta3 = finish3 - start3
    print(delta1, delta2, delta3)
    diff = np.linalg.norm(result_2 - result_3)
    print(diff)
