import numpy as np
import pandas as pd
import tqdm

def get_ic(feature_labels, compute_labels):
    """
    计算IC
    :param feature_labels: 特征标签
    :param compute_labels: 计算的标签
    :return: IC
    """
    ic = []
    for i in range(feature_labels.shape[1]):
        ic.append(np.corrcoef(feature_labels[:, i], compute_labels[:, i])[0, 1])
    return ic


def compute_label_from_pre_future_midprice_with_index(current_last_midprice, future_midprices, raw_time_intervals=[0.5, 1, 3, 5, 10, 20, 60, 120], time_index_factor=250, transform_func=None):
    """
    计算标签
    :param current_last_midprice: 当前最后一个中间价(需经过逆归一化)
    :param future_midprice: 未来中间价（需经过逆归一化）
    :param raw_time_interval: 原始时间间隔
    :param time_index_factor: 时间索引因子
    :return: 收益率标签
    """
    # assert current_last_midprice.shape[0] == future_midprices.shape[0], "batch size must be equal, but got {} and {}".format(current_last_midprice.shape[0], future_midprices.shape[0])
    future_pre_len = future_midprices.shape[1]
    
    raw_time_intervals = np.array(raw_time_intervals)
    next_time_index = raw_time_intervals * time_index_factor
    next_time_index = next_time_index.astype(int)
    next_time_index = np.clip(next_time_index, 0, future_pre_len-1)

    next_midprice = future_midprices[:, next_time_index]
    
    # apply transform function
    if transform_func is not None:
        next_midprice = transform_func(next_midprice)
        current_last_midprice = transform_func(current_last_midprice)
        
    pre_rate = next_midprice/current_last_midprice - 1
    
    return pre_rate


def compute_label_from_midprice_with_time_index_factor(midprice, raw_time_interval, time_index_factor):
    """
    计算标签
    :param midprice: 中间价
    :param raw_time_interval: 原始时间间隔
    :param time_index_factor: 时间索引因子
    :return: 标签
    """
    label = []
    for i in tqdm.tqdm(range(len(midprice))):
        # 找到下一个时间戳对应的中间价
        next_index = i + raw_time_interval*time_index_factor if (i + raw_time_interval*time_index_factor) < len(midprice) else len(midprice)-1
        next_index = int(next_index)
        next_mid_price = midprice[next_index]
        # 计算标签
        label.append((next_mid_price - midprice[i]) / midprice[i])
    return np.array(label)


def compute_label_with_index_interplot(timestamps, new_timestamps, interpolated_mid_price, label_time_interval_raw=[0.5, 1, 3, 5, 10, 20, 60, 120], time_index_factor=1000):
    """
    计算标签
    :param timestamps: 原始时间戳
    :param new_timestamps: 新的时间戳
    :param interpolated_mid_price: 插值后的价格
    :param label_time_interval_raw: 标签时间间隔
    :param time_index_factor: 时间索引因子
    :return: 标签
    """
    label_index_interval_raw = np.array(label_time_interval_raw) * time_index_factor
    label_index_interval_raw = label_index_interval_raw.astype(int)
    
    labels = []
    current_new_timestamp_index = 0
    for i in tqdm.tqdm(range(len(timestamps))):
        # 找到下一个时间戳
        current_timestamp = timestamps[i]
        while current_new_timestamp_index+1 < len(new_timestamps) and new_timestamps[current_new_timestamp_index] < current_timestamp:
            current_new_timestamp_index += 1
            
        current_interplot_mid_price = interpolated_mid_price[current_new_timestamp_index]
        next_new_timestamp_indexs = current_new_timestamp_index + label_index_interval_raw
        next_new_timestamp_indexs = np.clip(next_new_timestamp_indexs, 0, len(new_timestamps)-1)
        next_interplot_mid_price = interpolated_mid_price[next_new_timestamp_indexs]
        label = next_interplot_mid_price / current_interplot_mid_price - 1
        labels.append(label)
        
    return np.array(labels)