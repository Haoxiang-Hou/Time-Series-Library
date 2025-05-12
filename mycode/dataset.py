# 后续先做B3WIN和CMEES，这里先放了B3WIN的特征和原始行情，后续天的还在产，预计晚上产完
# /mnt/nas-intern/homes/zoezhu/data/data_B3WIN/

# 1. 其中*_feature.npy，是存了时间戳+15个特征+从短到长的八个label
# 2. *_snapshot.npy，存了原始行情，分别是'localtime'，'bp1'，'ap1'，'bv1'，'av1'，'lastprice'，'total_volume'，'total_turnover'

# B3WIN使用测试数据：[20240603,20240604,20240605,20240606,20240607,20240610,20240611,20240612,20240613,20240614,20240617,20240618,20240619,20240620,20240621,20240624,20240625]
# CMEES使用测试数据：[20250204, 20250205, 20250206, 20250207, 20250210, 20250211, 20250212]

import tqdm
import os
import glob
import json
import random

import numpy as np
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset


class FeatureSeries(Dataset):
    def __init__(self, data_paths, history_seq_len, batch_rolling_window_stride, padding_value, preprocess_stats_path):
        self.data_path = data_paths
        
        self.history_seq_len = history_seq_len
        self.batch_rolling_window_stride = batch_rolling_window_stride
        self.padding_value = padding_value
        
        self.preprocess_stats_path = preprocess_stats_path
        
        # load stats
        self.mean = np.load(os.path.join(self.preprocess_stats_path, 'mean.npy'))
        self.std = np.load(os.path.join(self.preprocess_stats_path, 'std.npy'))
        
        self.data_len_dict = json.load(open(os.path.join(self.preprocess_stats_path, 'stats.json'), 'r'))
        
        # calculate seq_len
        self.data_len_list = [self.data_len_dict[f] for f in self.data_path]
        self.seq_len_list = []
        for data_len in self.data_len_list:
            # after padding, the data length is data_len + history_seq_len - 1
            # after slicing, (seq_len - 1) * batch_rolling_window_stride + history_seq_len = data_len + history_seq_len - 1
            # seq_len = (data_len + history_seq_len - 1 - history_seq_len) / batch_rolling_window_stride + 1
            seq_len = (int((data_len - 1) / batch_rolling_window_stride) + 1)
            self.seq_len_list.append(seq_len)
            
        # calculate total seq_len
        self.seq_len = sum(self.seq_len_list)
        print(f"total seq_len: {self.seq_len}")
        print(f"seq_len_list: {self.seq_len_list}")
        
        # self.idx_dict = {}
        # for i in tqdm.tqdm(range(self.seq_len)):
        #     self.idx_dict[i] = self.get_idx(i)
        
        # load data init
        self._data_path_idx = -1
        self._data = None
        self.load_data(0)
        
    def get_idx(self, idx):
        # get the data path index
        data_path_idx = 0
        for i in range(len(self.seq_len_list)):
            if idx < self.seq_len_list[i]:
                data_path_idx = i
                break
            idx -= self.seq_len_list[i]
        return data_path_idx, idx
        
    def load_data(self, data_path_idx):
        if self._data_path_idx == data_path_idx:
            return self._data
        
        if data_path_idx >= len(self.data_path):
            raise IndexError("Index out of range")
        
        # load data
        self._data = np.load(self.data_path[data_path_idx]) # [data_len, feature_dim]
        # remove timestamp
        self._data = self._data[:, 1:]
        # convert to float32
        self._data = self._data.astype(np.float32)
        # normalize
        self._data = (self._data - self.mean) / self.std
        # padding
        self._data = np.pad(self._data, ((self.history_seq_len - 1, 0), (0, 0)), mode='constant', constant_values=self.padding_value)
        
        # update data path index
        self._data_path_idx = data_path_idx
        
        return self._data
            
    def __len__(self):
        return self.seq_len

    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        # slice data
        start_idx = data_idx * self.batch_rolling_window_stride
        end_idx = start_idx + self.history_seq_len
        features = self._data[start_idx:end_idx, :-8]
        labels = self._data[start_idx:end_idx, -8:]
        # convert to tensor
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        
        return features, labels
    
    
    

class FeaturePriceSeries(Dataset):
    def __init__(self, feature_paths, snapshot_paths, preprocess_stats_path, rolling_window_stride, padding_value, history_seq_len, history_label_len, future_pre_len, ):
        self.feature_paths = feature_paths
        self.snapshot_paths = snapshot_paths
        
        self.preprocess_feature_stats_path = os.path.join(preprocess_stats_path, 'feature')
        self.preprocess_snapshot_stats_path = os.path.join(preprocess_stats_path, 'snapshot')
        
        self.rolling_window_stride = rolling_window_stride
        self.padding_value = padding_value
        
        self.history_seq_len = history_seq_len
        self.history_label_len = history_label_len
        self.future_pre_len = future_pre_len
        
        assert self.history_seq_len >= self.history_label_len, "history_seq_len must be greater than or equal to history_label_len, but got history_seq_len: {}, history_label_len: {}".format(self.history_seq_len, self.history_label_len)
        
        
        # load stats
        self.feature_mean = np.load(os.path.join(self.preprocess_feature_stats_path, 'mean.npy'))
        self.feature_std = np.load(os.path.join(self.preprocess_feature_stats_path, 'std.npy'))
        self.snapshot_mean = np.load(os.path.join(self.preprocess_snapshot_stats_path, 'mean.npy'))
        self.snapshot_std = np.load(os.path.join(self.preprocess_snapshot_stats_path, 'std.npy'))
        
        self.mid_price_mean = np.mean(self.snapshot_mean[:2])
        self.mid_price_std = np.mean(self.snapshot_std[:2]) # use the mean of std of bp1 and ap1 approximately
        
        self.feature_data_len_dict = json.load(open(os.path.join(self.preprocess_feature_stats_path, 'stats.json'), 'r'))
        
        # calculate seq_len
        self.data_len_list = [self.feature_data_len_dict[f] for f in self.feature_paths]
        self.seq_len_list = []
        for data_len in self.data_len_list:
            # after padding, the data length is data_len + history_seq_len - 1
            # after slicing, (seq_len - 1) * rolling_window_stride + history_seq_len = data_len + history_seq_len - 1
            # seq_len = (data_len + history_seq_len - 1 - history_seq_len) / rolling_window_stride + 1
            seq_len = (int((data_len - 1) / self.rolling_window_stride) + 1)
            self.seq_len_list.append(seq_len)
            
        # calculate total seq_len
        self.seq_len = sum(self.seq_len_list)
        print(f"total seq_len: {self.seq_len}")
        print(f"seq_len_list: {self.seq_len_list}")
        
        # self.idx_dict = {}
        # for i in tqdm.tqdm(range(self.seq_len)):
        #     self.idx_dict[i] = self.get_idx(i)
        
        # load data init
        self._data_path_idx = -1
        self._feature_data = None
        self.load_data(0)
        
    def get_idx(self, idx):
        # get the data path index
        data_path_idx = 0
        for i in range(len(self.seq_len_list)):
            if idx < self.seq_len_list[i]:
                data_path_idx = i
                break
            idx -= self.seq_len_list[i]
        return data_path_idx, idx
        
    def load_data(self, data_path_idx):
        if self._data_path_idx == data_path_idx:
            return self._feature_data, self._snapshot_data
        
        if data_path_idx >= len(self.feature_paths):
            raise IndexError("Index out of range")
        
        # load feature data
        _feature_data = np.load(self.feature_paths[data_path_idx]) # [data_len, feature_dim]
        # load snapshot data
        _snapshot_data = np.load(self.snapshot_paths[data_path_idx]) # [data_len, snapshot_dim]
        
        assert _feature_data.shape[0] == _snapshot_data.shape[0], f"feature data shape: {_feature_data.shape}, snapshot data shape: {_snapshot_data.shape}"
        
        # # remove timestamp
        self._feature_data = _feature_data[:, 1:]
        self._snapshot_data = _snapshot_data[:, 1:]
        self._time_stamp = _snapshot_data[:, 0]
        # convert to float32
        self._feature_data = self._feature_data.astype(np.float32)
        self._snapshot_data = self._snapshot_data.astype(np.float32)
        self._time_stamp = self._time_stamp.astype(np.float32)
        # normalize
        self._feature_data = (self._feature_data - self.feature_mean) / self.feature_std
        self._snapshot_data = (self._snapshot_data - self.snapshot_mean) / self.snapshot_std
        # padding
        self._feature_data = np.pad(self._feature_data, ((self.history_seq_len - 1, self.future_pre_len), (0, 0)), mode='constant', constant_values=self.padding_value)
        self._snapshot_data = np.pad(self._snapshot_data, ((self.history_seq_len - 1, self.future_pre_len), (0, 0)), mode='edge')
        self._time_stamp = np.pad(self._time_stamp, ((self.history_seq_len - 1, self.future_pre_len),), mode='edge')
        
        # update data path index
        self._data_path_idx = data_path_idx
        
        return self._feature_data, self._snapshot_data, self._time_stamp
            
    def __len__(self):
        return self.seq_len

    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        # slice data
        feature_start_idx = data_idx * self.rolling_window_stride
        feature_end_idx = feature_start_idx + self.history_seq_len
        current_last_idx = feature_start_idx + self.history_seq_len - 1
        snapshot_start_idx = feature_end_idx - self.history_label_len
        snapshot_end_idx = snapshot_start_idx + self.history_label_len + self.future_pre_len
        
        features = self._feature_data[feature_start_idx:feature_end_idx, :15] # [history_seq_len, feature_dim]
        labels = self._feature_data[current_last_idx, -8:] # [8]
        mid_prices = self._snapshot_data[snapshot_start_idx:snapshot_end_idx, :2] # [history_label_len + future_pre_len, 2]
        history_mid_prices = mid_prices[:self.history_label_len].mean(axis=1) # [history_label_len]
        future_mid_prices = mid_prices[self.history_label_len:].mean(axis=1) # [future_pre_len]
        time_stamps = self._time_stamp[snapshot_start_idx:snapshot_end_idx] # [history_label_len + future_pre_len]
        history_time_stamps = time_stamps[:self.history_label_len]
        future_time_stamps = time_stamps[self.history_label_len:]
        
        # convert to tensor
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        history_mid_prices = torch.tensor(history_mid_prices)
        future_mid_prices = torch.tensor(future_mid_prices)
        history_time_stamps = torch.tensor(history_time_stamps)
        future_time_stamps = torch.tensor(future_time_stamps)
        
        return features, history_mid_prices, labels, future_mid_prices, history_time_stamps, future_time_stamps
    
    def inverse_transform_mid_price(self, mid_prices_normalized):
        mid_prices = mid_prices_normalized * self.mid_price_std + self.mid_price_mean
        return mid_prices
    
    
    
class FeatureInterpolatPriceSeries(Dataset):
    def __init__(self, data_dir_path, preprocess_stats_path, date_list, rolling_window_stride, padding_value, history_seq_len, history_label_len, future_pre_len, interpolat_freq_per_second=1000):
        self.feature_paths = [os.path.join(data_dir_path, f"{date}_feature.npy") for date in date_list]
        self.snapshot_paths = [os.path.join(data_dir_path, f"{date}_snapshot.npy") for date in date_list]
        self.interpolated_mid_prices_paths = [os.path.join(data_dir_path, f"{date}_interpolated_mid_prices.npy") for date in date_list]
        
        self.preprocess_feature_stats_path = os.path.join(preprocess_stats_path, 'feature')
        self.preprocess_snapshot_stats_path = os.path.join(preprocess_stats_path, 'snapshot')
        self.preprocess_interpolated_mid_prices_stats_path = os.path.join(preprocess_stats_path, 'interpolated_mid_prices')
        
        self.rolling_window_stride = rolling_window_stride
        self.padding_value = padding_value
        
        self.history_seq_len = history_seq_len
        self.history_label_len = history_label_len
        self.future_pre_len = future_pre_len
        
        self.interpolat_freq_per_second = interpolat_freq_per_second
        self.interpolate_interval = int(1 / self.interpolat_freq_per_second * 1e9)
        
        assert self.history_seq_len >= self.history_label_len, "history_seq_len must be greater than or equal to history_label_len, but got history_seq_len: {}, history_label_len: {}".format(self.history_seq_len, self.history_label_len)
        
        
        # load stats
        self.feature_mean = np.load(os.path.join(self.preprocess_feature_stats_path, 'mean.npy'))
        self.feature_std = np.load(os.path.join(self.preprocess_feature_stats_path, 'std.npy'))
        self.snapshot_mean = np.load(os.path.join(self.preprocess_snapshot_stats_path, 'mean.npy'))
        self.snapshot_std = np.load(os.path.join(self.preprocess_snapshot_stats_path, 'std.npy'))
        self.mid_price_mean = np.load(os.path.join(self.preprocess_interpolated_mid_prices_stats_path, 'mean.npy'))
        self.mid_price_std = np.load(os.path.join(self.preprocess_interpolated_mid_prices_stats_path, 'std.npy'))
        
        self.feature_data_len_dict = json.load(open(os.path.join(self.preprocess_feature_stats_path, 'stats.json'), 'r'))
        
        # calculate seq_len
        self.data_len_list = [self.feature_data_len_dict[f] for f in self.feature_paths]
        self.seq_len_list = []
        for data_len in self.data_len_list:
            # after padding, the data length is data_len + history_seq_len - 1
            # after slicing, (seq_len - 1) * rolling_window_stride + history_seq_len = data_len + history_seq_len - 1
            # seq_len = (data_len + history_seq_len - 1 - history_seq_len) / rolling_window_stride + 1
            seq_len = (int((data_len - 1) / self.rolling_window_stride) + 1)
            self.seq_len_list.append(seq_len)
            
        # calculate total seq_len
        self.seq_len = sum(self.seq_len_list)
        print(f"total seq_len: {self.seq_len}")
        print(f"seq_len_list: {self.seq_len_list}")
        
        # self.idx_dict = {}
        # for i in tqdm.tqdm(range(self.seq_len)):
        #     self.idx_dict[i] = self.get_idx(i)
        
        # load data init
        self._data_path_idx = -1
        self.load_data(0)
        
    def get_idx(self, idx):
        # get the data path index
        data_path_idx = 0
        for i in range(len(self.seq_len_list)):
            if idx < self.seq_len_list[i]:
                data_path_idx = i
                break
            idx -= self.seq_len_list[i]
        return data_path_idx, idx
        
    def load_data(self, data_path_idx):
        if self._data_path_idx == data_path_idx:
            return self._feature_data, self._snapshot_data
        
        if data_path_idx >= len(self.feature_paths):
            raise IndexError("Index out of range")
        
        # load data
        _feature_data = np.load(self.feature_paths[data_path_idx]) # [data_len, feature_dim]
        _snapshot_data = np.load(self.snapshot_paths[data_path_idx]) # [data_len, snapshot_dim]
        _interpolated_mid_prices = np.load(self.interpolated_mid_prices_paths[data_path_idx]) # [data_len, snapshot_dim]
        
        assert _feature_data.shape[0] == _snapshot_data.shape[0], f"feature data shape: {_feature_data.shape}, snapshot data shape: {_snapshot_data.shape}"
        
        # # remove timestamp
        self._feature_data = _feature_data[:, 1:]
        self._snapshot_data = _snapshot_data[:, 1:]
        self._time_stamps = _snapshot_data[:, 0]
        self._interpolated_time_stamps = _interpolated_mid_prices[:, 0]
        self._interpolated_mid_prices = _interpolated_mid_prices[:, 1]
        # convert to float32
        self._feature_data = self._feature_data.astype(np.float32)
        self._snapshot_data = self._snapshot_data.astype(np.float32)
        # self._time_stamps = self._time_stamps.astype(np.float32)
        # self._interpolated_time_stamps = self._interpolated_time_stamps.astype(np.float32)
        self._interpolated_mid_prices = self._interpolated_mid_prices.astype(np.float32)
        # normalize
        self._feature_data = (self._feature_data - self.feature_mean) / self.feature_std
        self._snapshot_data = (self._snapshot_data - self.snapshot_mean) / self.snapshot_std
        self._interpolated_mid_prices = (self._interpolated_mid_prices - self.mid_price_mean) / self.mid_price_std
        # padding
        self._feature_data = np.pad(self._feature_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='constant', constant_values=self.padding_value)
        self._snapshot_data = np.pad(self._snapshot_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='edge')
        self._time_stamps = np.pad(self._time_stamps, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        self._interpolated_time_stamps = np.pad(self._interpolated_time_stamps, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        self._interpolated_mid_prices = np.pad(self._interpolated_mid_prices, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        
        # update data path index
        self._data_path_idx = data_path_idx
        
        return self._feature_data, self._snapshot_data, self._time_stamps, self._interpolated_time_stamps, self._interpolated_mid_prices
            
    def __len__(self):
        return self.seq_len

    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        # slice data
        feature_start_idx = data_idx * self.rolling_window_stride
        feature_end_idx = feature_start_idx + self.history_seq_len
        
        current_last_idx = feature_start_idx + self.history_seq_len - 1
        current_timestamp = self._time_stamps[current_last_idx]
        interploted_current_idx = int(self.history_seq_len + (current_timestamp - self._time_stamps[0]) // self.interpolate_interval)
        interploted_current_idx = min(interploted_current_idx, len(self._interpolated_time_stamps) - self.future_pre_len - 1)
        
        # interploted_current_idx = np.searchsorted(self._interpolated_time_stamps, current_timestamp)
        # interploted_current_idx = np.clip(interploted_current_idx, self.history_seq_len, len(self._interpolated_time_stamps) - self.future_pre_len - 1)
        
        interploted_start_idx = interploted_current_idx - self.history_label_len
        interploted_end_idx = interploted_current_idx + self.future_pre_len
        
        features = self._feature_data[feature_start_idx:feature_end_idx, :15] # [history_seq_len, feature_dim]
        labels = self._feature_data[current_last_idx, -8:] # [8]
        
        history_interpolated_mid_prices = self._interpolated_mid_prices[interploted_start_idx:interploted_current_idx] # [history_label_len]
        future_interpolated_mid_prices = self._interpolated_mid_prices[interploted_current_idx:interploted_end_idx] # [future_pre_len]
        
        history_interpolated_time_stamps = self._interpolated_time_stamps[interploted_start_idx:interploted_current_idx] # [history_label_len]
        future_interpolated_time_stamps = self._interpolated_time_stamps[interploted_current_idx:interploted_end_idx] # [future_pre_len]
        
        # convert to tensor
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        
        history_interpolated_mid_prices = torch.tensor(history_interpolated_mid_prices)
        future_interpolated_mid_prices = torch.tensor(future_interpolated_mid_prices)
        
        history_interpolated_time_stamps = torch.tensor(history_interpolated_time_stamps)
        future_interpolated_time_stamps = torch.tensor(future_interpolated_time_stamps)
        
        return features, history_interpolated_mid_prices, history_interpolated_time_stamps, labels, future_interpolated_mid_prices, future_interpolated_time_stamps
    
    
    def inverse_transform_mid_price(self, mid_prices_normalized):
        mid_prices = mid_prices_normalized * self.mid_price_std + self.mid_price_mean
        return mid_prices
    
    
    def interpolate_mid_price(self, mid_prices, time_stamps):
        # 检查并移除重复值
        unique_indices = np.unique(time_stamps, return_index=True)[1]
        time_stamps = time_stamps[unique_indices]
        mid_prices = mid_prices[unique_indices]

        # 检查并移除 NaN 或 inf
        valid_indices = ~np.isnan(time_stamps) & ~np.isnan(mid_prices) & ~np.isinf(time_stamps) & ~np.isinf(mid_prices)
        time_stamps = time_stamps[valid_indices]
        mid_prices = mid_prices[valid_indices]

        # 计算插值后的 time_stamps, 单位ns
        
        start_time_stamp = time_stamps[0]
        end_time_stamp = start_time_stamp + self.interpolate_interval * (self.future_pre_len-1) 
        
        if time_stamps[-1] < end_time_stamp:
            # 如果时间间隔小于插值间隔，则直接返回
            # print(f"time_stamps[-1] < end_time_stamp, end_time_stamp-time_stamps[-1]: {end_time_stamp-time_stamps[-1]}")
            time_stamps_interpolate, mid_prices_interpolate = time_stamps, mid_prices
        else:
            interp_func = interp1d(time_stamps, mid_prices, kind='linear', fill_value="extrapolate")
            
            time_stamps_interpolate = np.arange(start_time_stamp, end_time_stamp, self.interpolate_interval)
            mid_prices_interpolate = interp_func(time_stamps_interpolate)
        
        # change len to future_pre_len
        if len(mid_prices_interpolate) > self.future_pre_len:
            time_stamps_interpolate = time_stamps_interpolate[:self.future_pre_len]
            mid_prices_interpolate = mid_prices_interpolate[:self.future_pre_len]
        elif len(mid_prices_interpolate) < self.future_pre_len:
            time_stamps_interpolate = np.pad(time_stamps_interpolate, (0, self.future_pre_len - len(mid_prices_interpolate)), mode='edge')
            mid_prices_interpolate = np.pad(mid_prices_interpolate, (0, self.future_pre_len - len(mid_prices_interpolate)), mode='edge')
        
        return time_stamps_interpolate, mid_prices_interpolate
    
    

class FeatureLastHistoryMidPriceSeries(Dataset):
    def __init__(self, data_dir_path, preprocess_stats_path, date_list, rolling_window_stride, padding_value, history_seq_len, history_label_len, future_pre_len, freq_per_second=100):
        self.feature_paths = [os.path.join(data_dir_path, f"{date}_feature.npy") for date in date_list]
        self.snapshot_paths = [os.path.join(data_dir_path, f"{date}_snapshot.npy") for date in date_list]
        
        self.preprocess_feature_stats_path = os.path.join(preprocess_stats_path, 'feature')
        self.preprocess_snapshot_stats_path = os.path.join(preprocess_stats_path, 'snapshot')
        self.preprocess_interpolated_mid_prices_stats_path = os.path.join(preprocess_stats_path, 'interpolated_mid_prices')
        
        self.rolling_window_stride = rolling_window_stride
        self.padding_value = padding_value
        
        self.history_seq_len = history_seq_len
        self.history_label_len = history_label_len
        self.future_pre_len = future_pre_len
        
        self.freq_per_second = freq_per_second
        self.time_interval = int(1e9 / self.freq_per_second)
        
        assert self.history_seq_len >= self.history_label_len, "history_seq_len must be greater than or equal to history_label_len, but got history_seq_len: {}, history_label_len: {}".format(self.history_seq_len, self.history_label_len)
        
        
        # load stats
        self.feature_mean = np.load(os.path.join(self.preprocess_feature_stats_path, 'mean.npy'))
        self.feature_std = np.load(os.path.join(self.preprocess_feature_stats_path, 'std.npy'))
        self.snapshot_mean = np.load(os.path.join(self.preprocess_snapshot_stats_path, 'mean.npy'))
        self.snapshot_std = np.load(os.path.join(self.preprocess_snapshot_stats_path, 'std.npy'))
        self.mid_price_mean = np.load(os.path.join(self.preprocess_interpolated_mid_prices_stats_path, 'mean.npy'))
        self.mid_price_std = np.load(os.path.join(self.preprocess_interpolated_mid_prices_stats_path, 'std.npy'))
        
        self.feature_data_len_dict = json.load(open(os.path.join(self.preprocess_feature_stats_path, 'stats.json'), 'r'))
        
        # calculate seq_len
        self.data_len_list = [self.feature_data_len_dict[f] for f in self.feature_paths]
        self.seq_len_list = []
        for data_len in self.data_len_list:
            # after padding, the data length is data_len + history_seq_len - 1
            # after slicing, (seq_len - 1) * rolling_window_stride + history_seq_len = data_len + history_seq_len - 1
            # seq_len = (data_len + history_seq_len - 1 - history_seq_len) / rolling_window_stride + 1
            seq_len = (int((data_len - 1) / self.rolling_window_stride) + 1)
            self.seq_len_list.append(seq_len)
            
        # calculate total seq_len
        self.seq_len = sum(self.seq_len_list)
        print(f"total seq_len: {self.seq_len}")
        print(f"seq_len_list: {self.seq_len_list}")
        
        # self.idx_dict = {}
        # for i in tqdm.tqdm(range(self.seq_len)):
        #     self.idx_dict[i] = self.get_idx(i)
        
        # load data init
        self._data_path_idx = -1
        self.load_data(0)
        
    def get_idx(self, idx):
        # get the data path index
        data_path_idx = 0
        for i in range(len(self.seq_len_list)):
            if idx < self.seq_len_list[i]:
                data_path_idx = i
                break
            idx -= self.seq_len_list[i]
        return data_path_idx, idx
        
    def load_data(self, data_path_idx):
        if self._data_path_idx == data_path_idx:
            return self._feature_data, self._snapshot_data
        
        if data_path_idx >= len(self.feature_paths):
            raise IndexError("Index out of range")
        
        # load data
        _feature_data = np.load(self.feature_paths[data_path_idx]) # [data_len, feature_dim]
        _snapshot_data = np.load(self.snapshot_paths[data_path_idx]) # [data_len, snapshot_dim]
        
        assert _feature_data.shape[0] == _snapshot_data.shape[0], f"feature data shape: {_feature_data.shape}, snapshot data shape: {_snapshot_data.shape}"
        
        # remove timestamp
        self._feature_data = _feature_data[:, 1:]
        self._snapshot_data = _snapshot_data[:, 1:]
        self._time_stamps = _snapshot_data[:, 0]
        self._mid_prices = _snapshot_data[:, 1:3].mean(axis=1) # [data_len]
        # convert to float32
        self._feature_data = self._feature_data.astype(np.float32)
        self._snapshot_data = self._snapshot_data.astype(np.float32)
        # self._time_stamps = self._time_stamps.astype(np.float32) # float32 is not enough for nanosecond
        self._mid_prices = self._mid_prices.astype(np.float32)
        # normalize
        self._feature_data = (self._feature_data - self.feature_mean) / self.feature_std
        # self._snapshot_data = (self._snapshot_data - self.snapshot_mean) / self.snapshot_std
        self._mid_prices = (self._mid_prices - self.mid_price_mean) / self.mid_price_std
        # padding
        # self._feature_data = np.pad(self._feature_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='constant', constant_values=self.padding_value)
        # # self._snapshot_data = np.pad(self._snapshot_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='edge')
        # self._time_stamps = np.pad(self._time_stamps, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        # self._mid_prices = np.pad(self._mid_prices, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        
        # update data path index
        self._data_path_idx = data_path_idx
        
        return self._feature_data, self._snapshot_data, self._time_stamps, self._mid_prices
            
    def __len__(self):
        return self.seq_len

    # def __getitem__(self, idx):
    #     data_path_idx, data_idx = self.get_idx(idx)
    #     self.load_data(data_path_idx)
            
    #     current_idx = data_idx * self.rolling_window_stride
        
    #     # select by last history time stamp 
    #     future_idx_list = []
    #     _idx = current_idx
    #     for i in range(1, self.future_pre_len+1):
    #         while _idx+1 < len(self._time_stamps) and self._time_stamps[_idx+1] <= self._time_stamps[current_idx] + self.time_interval*i:
    #             _idx += 1
    #         future_idx_list.append(_idx)
        
    #     history_idx_list = []
    #     _idx = current_idx
    #     for i in range(self.history_seq_len):
    #         while _idx-1 >= 0 and self._time_stamps[_idx] > self._time_stamps[current_idx] - self.time_interval*i:
    #             _idx -= 1
    #         history_idx_list.append(_idx) 
    #     history_idx_list = history_idx_list[::-1]
        
    #     # select data idx lists
    #     history_feature_seq_idx_list = history_idx_list 
    #     history_label_idx_list = history_idx_list[-self.history_label_len:] # [history_label_len]
        
    #     # self.history_seq_len = history_seq_len
    #     # self.history_label_len = history_label_len
    #     # self.future_pre_len = future_pre_len
        
    #     features = self._feature_data[history_feature_seq_idx_list, :15] # [history_seq_len, feature_dim]
    #     labels = self._feature_data[current_idx, -8:] # [8]
    #     history_mid_prices = self._mid_prices[history_label_idx_list] # [history_label_len]
    #     future_mid_prices = self._mid_prices[future_idx_list] # [future_pre_len]
    #     history_time_stamps = self._time_stamps[history_label_idx_list] # [history_label_len]
    #     future_time_stamps = self._time_stamps[future_idx_list] # [future_pre_len]
        
    #     # convert to tensor
    #     features = torch.tensor(features)
    #     labels = torch.tensor(labels)
    #     history_mid_prices = torch.tensor(history_mid_prices)
    #     future_mid_prices = torch.tensor(future_mid_prices)
    #     history_time_stamps = torch.tensor(history_time_stamps)
    #     future_time_stamps = torch.tensor(future_time_stamps)
        
    #     # print(f"history_time_stamps: {history_time_stamps[:10]-history_time_stamps[0]}\n, future_time_stamps: {future_time_stamps[:10]-future_time_stamps[0]}")
        
    #     return features, history_mid_prices, history_time_stamps, labels, future_mid_prices, future_time_stamps
    
    
    def inverse_transform_mid_price(self, mid_prices_normalized):
        mid_prices = mid_prices_normalized * self.mid_price_std + self.mid_price_mean
        return mid_prices
    