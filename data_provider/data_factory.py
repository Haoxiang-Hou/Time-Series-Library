import mycode.dataset as dataset

import os
import glob

import numpy as np
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from mycode import ic_compute

class FeatureDataset(dataset.FeatureSeries):
    def __init__(self, data_dir_path, preprocess_stats_path, date_list, rolling_window_stride, padding_value, history_seq_len, history_label_len, future_pre_len, freq_per_second):
        self.rolling_window_stride = rolling_window_stride
        self.x_seq_len = history_seq_len
        self.y_label_len = history_label_len
        self.y_pred_len = future_pre_len
        assert future_pre_len == 1, f"future_pre_len should be 1, but got {future_pre_len}"
        
        data_paths = [os.path.join(data_dir_path, f"{date}_feature.npy") for date in date_list]
        batch_rolling_window_stride = rolling_window_stride
        preprocess_stats_path = os.path.join(preprocess_stats_path, 'feature')
        super().__init__(data_paths, history_seq_len, batch_rolling_window_stride, padding_value, preprocess_stats_path)
        
    
    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        # slice data
        start_idx = data_idx * self.batch_rolling_window_stride
        end_idx = start_idx + self.history_seq_len
        features = self._data[start_idx:end_idx, :-8] # [history_seq_len, feature_dim]
        labels = self._data[end_idx, -8:] # [8]
        # convert to tensor
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        
        padded_labels = torch.cat((torch.zeros(7), labels), dim=0).unsqueeze(0) # [1, 8]
        
        seq_x = features
        # seq_y = features[-self.y_label_len:]
        seq_y = seq_y = torch.cat((features[-self.y_label_len:], padded_labels), dim=0) # [history_label_len + future_pre_len, 15]
        
        return seq_x, seq_y, torch.tensor(0), torch.tensor(0), labels

class timeSeriesDataset(dataset.FeatureInterpolatPriceSeries):
    def __init__(self, data_dir_path, preprocess_stats_path, date_list, rolling_window_stride, padding_value, history_seq_len, history_label_len, future_pre_len, freq_per_second):
        self.x_seq_len = history_seq_len
        self.y_label_len = history_label_len
        self.y_pred_len = future_pre_len
        
        self.freq_per_second = freq_per_second
        assert 1000 % self.freq_per_second == 0, f"freq_per_second should be a divisor of 1000, but got {self.freq_per_second}"
        self.sampled_freq = 1000 // self.freq_per_second
        
        super().__init__(data_dir_path=data_dir_path,
                         preprocess_stats_path=preprocess_stats_path,
                         date_list=date_list,
                         rolling_window_stride=rolling_window_stride,
                         padding_value=padding_value,
                         history_seq_len=history_seq_len*self.sampled_freq,
                         history_label_len=history_seq_len*self.sampled_freq, # use history_seq_len of mid prices
                         future_pre_len=future_pre_len*self.sampled_freq,
                         interpolat_freq_per_second=1000)
        
        
    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        # slice data
        feature_start_idx = data_idx * self.rolling_window_stride
        
        current_last_idx = feature_start_idx + self.history_seq_len - 1
        current_timestamp = self._time_stamps[current_last_idx]
        interploted_current_idx = int(self.history_seq_len + (current_timestamp - self._time_stamps[0]) // self.interpolate_interval)
        interploted_current_idx = min(interploted_current_idx, len(self._interpolated_time_stamps) - self.future_pre_len - 1)
        
        interploted_start_idx = interploted_current_idx - self.history_label_len
        interploted_end_idx = interploted_current_idx + self.future_pre_len
        
        
        interploted_label_start_idx = interploted_current_idx - self.y_label_len * self.sampled_freq
        
        seq_x = self._interpolated_mid_prices[interploted_start_idx:interploted_current_idx:self.sampled_freq] # [history_label_len]
        seq_y = self._interpolated_mid_prices[interploted_label_start_idx:interploted_end_idx:self.sampled_freq] # [history_label_len + future_pre_len]
        seq_x_mark = self._interpolated_time_stamps[interploted_start_idx:interploted_current_idx:self.sampled_freq]
        seq_y_mark = self._interpolated_time_stamps[interploted_label_start_idx:interploted_end_idx:self.sampled_freq]
        
        labels = self._feature_data[current_last_idx, -8:] # [8]
        
        # assert len(seq_x) == self.x_seq_len, f"seq_x length: {len(seq_x)}, expected: {self.x_seq_len}"
        # assert len(seq_y) == self.y_label_len + self.y_pred_len, f"seq_y length: {len(seq_y)}, expected: {self.y_label_len + self.y_pred_len}"
            
        seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1)
        seq_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32).unsqueeze(-1)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32).unsqueeze(-1)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, labels
    
    def inverse_transform(self, data):
        return super().inverse_transform_mid_price(data)
    



class LastHistoryMidPriceSeriesDataset(dataset.FeatureLastHistoryMidPriceSeries):
    def __init__(self, data_dir_path, preprocess_stats_path, date_list, rolling_window_stride, padding_value, history_seq_len=96, history_label_len=48, future_pre_len=96, freq_per_second=100):
        self.x_seq_len = history_seq_len
        self.y_label_len = history_label_len
        self.y_pred_len = future_pre_len
        
        self.freq_per_second = freq_per_second
        # assert 1000 % self.freq_per_second == 0, f"freq_per_second should be a divisor of 1000, but got {self.freq_per_second}"
        self.sampled_freq = 1
        
        
        super().__init__(data_dir_path=data_dir_path,
                         preprocess_stats_path=preprocess_stats_path,
                         date_list=date_list,
                         rolling_window_stride=rolling_window_stride,
                         padding_value=padding_value,
                         history_seq_len=history_seq_len*self.sampled_freq,
                         history_label_len=history_seq_len*self.sampled_freq, # use history_seq_len of mid prices
                         future_pre_len=future_pre_len*self.sampled_freq,
                         freq_per_second=freq_per_second)
        
        
    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        current_idx = data_idx * self.rolling_window_stride
        
        # select by last history time stamp 
        future_idx_list = []
        _idx = current_idx
        for i in range(1, self.future_pre_len+1):
            _expect_next_time = self._time_stamps[current_idx] + self.time_interval*i
            while _idx+1 < len(self._time_stamps) and self._time_stamps[_idx] <= _expect_next_time:
                _idx += 1
            future_idx_list.append(_idx)
        
        history_idx_list = []
        _idx = current_idx
        for i in range(self.history_seq_len):
            _expect_next_time = self._time_stamps[current_idx] - self.time_interval*i
            while _idx-1 >= 0 and self._time_stamps[_idx] > _expect_next_time:
                _idx -= 1
            history_idx_list.append(_idx) 
        history_idx_list = history_idx_list[::-1]

        # select data idx lists
        # history_feature_seq_idx_list = history_idx_list 
        history_label_idx_list = history_idx_list[-self.history_label_len:] # [history_label_len]
        
        # features = self._feature_data[history_feature_seq_idx_list, :15] # [history_seq_len, feature_dim]
        labels = self._feature_data[current_idx, -8:] # [8]
        history_mid_prices = self._mid_prices[history_label_idx_list] # [history_label_len]
        future_mid_prices = self._mid_prices[future_idx_list] # [future_pre_len]
        history_time_stamps = self._time_stamps[history_label_idx_list] # [history_label_len]
        future_time_stamps = self._time_stamps[future_idx_list] # [future_pre_len]
        
        history_mid_prices = torch.tensor(history_mid_prices, dtype=torch.float32)
        future_mid_prices = torch.tensor(future_mid_prices, dtype=torch.float32)
        history_time_stamps = torch.tensor(history_time_stamps, dtype=torch.float32)
        future_time_stamps = torch.tensor(future_time_stamps, dtype=torch.float32)
        
        seq_x = history_mid_prices
        seq_y = torch.cat((history_mid_prices[-self.y_label_len:], future_mid_prices), dim=0) # [history_label_len + future_pre_len, 15]
        seq_x_mark = history_time_stamps
        seq_y_mark = torch.cat((history_time_stamps[-self.y_label_len:], future_time_stamps), dim=0) # [history_label_len + future_pre_len, 15]
        
        # assert len(seq_x) == self.x_seq_len, f"seq_x length: {len(seq_x)}, expected: {self.x_seq_len}"
        # assert len(seq_y) == self.y_label_len + self.y_pred_len, f"seq_y length: {len(seq_y)}, expected: {self.y_label_len + self.y_pred_len}"
        # assert len(seq_x_mark) == self.x_seq_len, f"seq_x_mark length: {len(seq_x_mark)}, expected: {self.x_seq_len}"
        # assert len(seq_y_mark) == self.y_label_len + self.y_pred_len, f"seq_y_mark length: {len(seq_y_mark)}, expected: {self.y_label_len + self.y_pred_len}"
            
        seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1)
        seq_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32).unsqueeze(-1)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(labels)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, labels
    
    def inverse_transform(self, data):
        return super().inverse_transform_mid_price(data)
    
    

class SelectPriceFeaturesSeriesDataset(dataset.FeatureLastHistoryMidPriceSeries):
    def __init__(self, data_dir_path, preprocess_stats_path, date_list, rolling_window_stride, padding_value, history_seq_len=96, history_label_len=48, future_pre_len=96, freq_per_second=100):
        self.x_seq_len = history_seq_len
        self.y_label_len = history_label_len
        self.y_pred_len = future_pre_len
        
        self.freq_per_second = freq_per_second
        # assert 1000 % self.freq_per_second == 0, f"freq_per_second should be a divisor of 1000, but got {self.freq_per_second}"
        self.sampled_freq = 1
        
        
        super().__init__(data_dir_path=data_dir_path,
                         preprocess_stats_path=preprocess_stats_path,
                         date_list=date_list,
                         rolling_window_stride=rolling_window_stride,
                         padding_value=padding_value,
                         history_seq_len=history_seq_len*self.sampled_freq,
                         history_label_len=history_seq_len*self.sampled_freq, # use history_seq_len of mid prices
                         future_pre_len=future_pre_len*self.sampled_freq,
                         freq_per_second=freq_per_second)
        
        
    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        current_idx = data_idx * self.rolling_window_stride
        
        # select by last history time stamp 
        future_idx_list = []
        _idx = current_idx
        for i in range(1, self.future_pre_len+1):
            _expect_next_time = self._time_stamps[current_idx] + self.time_interval*i
            while _idx+1 < len(self._time_stamps) and self._time_stamps[_idx] <= _expect_next_time:
                _idx += 1
            future_idx_list.append(_idx)
        
        history_idx_list = []
        _idx = current_idx
        for i in range(self.history_seq_len):
            _expect_next_time = self._time_stamps[current_idx] - self.time_interval*i
            while _idx-1 >= 0 and self._time_stamps[_idx] > _expect_next_time:
                _idx -= 1
            history_idx_list.append(_idx) 
        history_idx_list = history_idx_list[::-1]

        # select data idx lists
        history_label_idx_list = history_idx_list[-self.history_label_len:] # [history_label_len]
        
        features = self._feature_data[history_label_idx_list, :15] # [history_label_len, 15]
        labels = self._feature_data[current_idx, -8:] # [8]
        history_mid_prices = self._mid_prices[history_label_idx_list] # [history_label_len]
        future_mid_prices = self._mid_prices[future_idx_list] # [future_pre_len]
        history_time_stamps = self._time_stamps[history_label_idx_list] # [history_label_len]
        future_time_stamps = self._time_stamps[future_idx_list] # [future_pre_len]
        
        features = torch.tensor(features, dtype=torch.float32) 
        history_mid_prices = torch.tensor(history_mid_prices, dtype=torch.float32)
        future_mid_prices = torch.tensor(future_mid_prices, dtype=torch.float32)
        history_time_stamps = torch.tensor(history_time_stamps, dtype=torch.float32)
        future_time_stamps = torch.tensor(future_time_stamps, dtype=torch.float32)
        
        history_features_midprices = torch.cat((features, history_mid_prices.unsqueeze(-1)), dim=-1) # [history_label_len, 16]
        future_features_midprices = torch.cat((torch.zeros((self.future_pre_len, 15)), future_mid_prices.unsqueeze(-1)), dim=-1) # [future_pre_len, 16]
        
        seq_x = history_features_midprices
        seq_y = torch.cat((history_features_midprices[-self.y_label_len:], future_features_midprices), dim=0) # [history_label_len + future_pre_len, 15]
        seq_x_mark = history_time_stamps
        seq_y_mark = torch.cat((history_time_stamps[-self.y_label_len:], future_time_stamps), dim=0) # [history_label_len + future_pre_len, 15]
        
        # assert len(seq_x) == self.x_seq_len, f"seq_x length: {len(seq_x)}, expected: {self.x_seq_len}"
        # assert len(seq_y) == self.y_label_len + self.y_pred_len, f"seq_y length: {len(seq_y)}, expected: {self.y_label_len + self.y_pred_len}"
        # assert len(seq_x_mark) == self.x_seq_len, f"seq_x_mark length: {len(seq_x_mark)}, expected: {self.x_seq_len}"
        # assert len(seq_y_mark) == self.y_label_len + self.y_pred_len, f"seq_y_mark length: {len(seq_y_mark)}, expected: {self.y_label_len + self.y_pred_len}"
            
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32).unsqueeze(-1)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(labels)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, labels
    
    def inverse_transform(self, data):
        return super().inverse_transform_mid_price(data)
    
    def get_pre_label(self, batch_x, future_mid_prices):
        current_last_midprice = batch_x[:, -1, -1].unsqueeze(-1).detach().cpu() # shape: [batch_size, 1]
        current_and_future_midprices = torch.cat([current_last_midprice, future_mid_prices.squeeze(-1)], dim=1) # shape: [batch_size, pred_len+1]
                
        return_label = ic_compute.compute_label_from_pre_future_midprice_with_index(current_last_midprice, current_and_future_midprices, raw_time_intervals=[0.5, 1, 3, 5, 10, 20, 60, 120], time_index_factor=self.freq_per_second, transform_func=self.inverse_transform)
        
        return return_label
        
        
        


class SelectReturnFeaturesSeriesDataset(dataset.FeatureLastHistoryMidPriceSeries):
    def __init__(self, data_dir_path, preprocess_stats_path, date_list, rolling_window_stride, padding_value, history_seq_len=96, history_label_len=48, future_pre_len=96, freq_per_second=100):
        self.x_seq_len = history_seq_len
        self.y_label_len = history_label_len
        self.y_pred_len = future_pre_len
        
        self.freq_per_second = freq_per_second
        # assert 1000 % self.freq_per_second == 0, f"freq_per_second should be a divisor of 1000, but got {self.freq_per_second}"
        self.sampled_freq = 1

        raw_time_intervals = [0.5, 1, 3, 5, 10, 20, 60, 120]
        raw_time_intervals = np.array(raw_time_intervals)
        next_time_index = raw_time_intervals * self.freq_per_second - 1
        next_time_index = next_time_index.astype(int)
        next_time_index = np.clip(next_time_index, 0, self.y_pred_len-1)
        self.next_time_index = next_time_index
        
        super().__init__(data_dir_path=data_dir_path,
                         preprocess_stats_path=preprocess_stats_path,
                         date_list=date_list,
                         rolling_window_stride=rolling_window_stride,
                         padding_value=padding_value,
                         history_seq_len=history_seq_len*self.sampled_freq,
                         history_label_len=history_seq_len*self.sampled_freq, # use history_seq_len of mid prices
                         future_pre_len=future_pre_len*self.sampled_freq,
                         freq_per_second=freq_per_second)
        
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
        # self._mid_prices = (self._mid_prices - self.mid_price_mean) / self.mid_price_std
        
        # padding
        # self._feature_data = np.pad(self._feature_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='constant', constant_values=self.padding_value)
        # # self._snapshot_data = np.pad(self._snapshot_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='edge')
        # self._time_stamps = np.pad(self._time_stamps, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        # self._mid_prices = np.pad(self._mid_prices, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        
        # update data path index
        self._data_path_idx = data_path_idx
        
        return self._feature_data, self._snapshot_data, self._time_stamps, self._mid_prices
        
    def __getitem__(self, idx):
        data_path_idx, data_idx = self.get_idx(idx)
        self.load_data(data_path_idx)
            
        current_idx = data_idx * self.rolling_window_stride
        
        # select by last history time stamp 
        future_idx_list = []
        _idx = current_idx
        for i in range(1, self.future_pre_len+1):
            _expect_next_time = self._time_stamps[current_idx] + self.time_interval*i
            while _idx+1 < len(self._time_stamps) and self._time_stamps[_idx] <= _expect_next_time:
                _idx += 1
            future_idx_list.append(_idx)
        
        history_idx_list = []
        _idx = current_idx
        for i in range(self.history_seq_len):
            _expect_next_time = self._time_stamps[current_idx] - self.time_interval*i
            while _idx-1 >= 0 and self._time_stamps[_idx] > _expect_next_time:
                _idx -= 1
            history_idx_list.append(_idx) 
        history_idx_list = history_idx_list[::-1]

        # select data idx lists
        history_label_idx_list = history_idx_list[-self.history_label_len:] # [history_label_len]
        
        features = self._feature_data[history_label_idx_list, :15] # [history_label_len, 15]
        labels = self._feature_data[current_idx, -8:] # [8]
        
        current_mid_prices = self._mid_prices[current_idx] # [1]
        current_time_stamps = self._time_stamps[current_idx] # [1]
        history_mid_prices = self._mid_prices[history_label_idx_list] # [history_label_len]
        future_mid_prices = self._mid_prices[future_idx_list] # [future_pre_len]
        history_time_stamps = self._time_stamps[history_label_idx_list] # [history_label_len]
        # future_time_stamps = self._time_stamps[future_idx_list] # [future_pre_len]
        _hyp_future_time_stamps = torch.arange(current_time_stamps, current_time_stamps + self.future_pre_len * self.time_interval, self.time_interval).float()
        future_time_stamps = _hyp_future_time_stamps
        
        history_returns = (history_mid_prices - current_mid_prices) / current_mid_prices
        future_returns = (future_mid_prices - current_mid_prices) / current_mid_prices
        
        # normalize
        history_returns = (history_returns - self.feature_mean[-8]) / self.feature_std[-8]
        future_returns = (future_returns - self.feature_mean[-8]) / self.feature_std[-8]
        
        features = torch.tensor(features, dtype=torch.float32) 
        history_returns = torch.tensor(history_returns, dtype=torch.float32)
        future_returns = torch.tensor(future_returns, dtype=torch.float32)
        history_time_stamps = torch.tensor(history_time_stamps, dtype=torch.float32)
        future_time_stamps = torch.tensor(future_time_stamps, dtype=torch.float32)
        
        history_features_return = torch.cat((features, history_returns.unsqueeze(-1)), dim=-1) # [history_label_len, 16]
        future_features_return = torch.cat((torch.zeros((self.future_pre_len, 15)), future_returns.unsqueeze(-1)), dim=-1) # [future_pre_len, 16]
        
        seq_x = history_features_return
        seq_y = torch.cat((history_features_return[-self.y_label_len:], future_features_return), dim=0) # [history_label_len + future_pre_len, 15]
        seq_x_mark = history_time_stamps
        seq_y_mark = torch.cat((history_time_stamps[-self.y_label_len:], future_time_stamps), dim=0) # [history_label_len + future_pre_len, 15]
        
        # assert len(seq_x) == self.x_seq_len, f"seq_x length: {len(seq_x)}, expected: {self.x_seq_len}"
        # assert len(seq_y) == self.y_label_len + self.y_pred_len, f"seq_y length: {len(seq_y)}, expected: {self.y_label_len + self.y_pred_len}"
        # assert len(seq_x_mark) == self.x_seq_len, f"seq_x_mark length: {len(seq_x_mark)}, expected: {self.x_seq_len}"
        # assert len(seq_y_mark) == self.y_label_len + self.y_pred_len, f"seq_y_mark length: {len(seq_y_mark)}, expected: {self.y_label_len + self.y_pred_len}"
            
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32).unsqueeze(-1)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32).unsqueeze(-1)
        labels = torch.tensor(labels)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, labels
    
    # def inverse_transform(self, data):
    #     return super().inverse_transform_mid_price(data)
    
    def get_pre_label(self, batch_x, future_returns):
        future_returns = future_returns.squeeze(-1)
        return_label = future_returns[:, self.next_time_index]
        return return_label
    
    

data_dict = {
    'FeatureDataset': FeatureDataset,
    'timeSeriesDataset': timeSeriesDataset,
    'LastHistoryMidPriceSeriesDataset': LastHistoryMidPriceSeriesDataset,
    'SelectPriceFeaturesSeriesDataset': SelectPriceFeaturesSeriesDataset,
    'SelectReturnFeaturesSeriesDataset': SelectReturnFeaturesSeriesDataset
}
    

def data_provider(args, flag):
    args.data = 'CMEES'
    args.data_path = './data/'
    args.stats_path = './my_exp/stats/CMEES'
    # args.rolling_window_stride = 100
    if flag == 'train':
        args.rolling_window_stride = args.train_rolling_window_stride
    if flag == 'val':
        args.rolling_window_stride = args.dev_rolling_window_stride
    if flag == 'test':
        args.rolling_window_stride = 1
    
    args.history_seq_len = args.seq_len
    args.history_label_len = args.label_len
    args.future_pre_len = args.pred_len
    # args.freq_per_second
    
    if args.data == 'B3WIN':
        test_dates = ['20240528', '20240603', '20240604', '20240605', '20240606', '20240607', '20240610', '20240611', '20240612', '20240613', '20240614', '20240617', '20240618', '20240619', '20240620', '20240621', '20240624', '20240625']
    elif args.data == 'CMEES':
        test_dates = ['20250204', '20250205', '20250206', '20250207', '20250210', '20250211', '20250212']
        # test_dates = ['20250212']

        
    data_dir_path = os.path.join(args.data_path, f"data_{args.data}")
    feature_paths = glob.glob(os.path.join(args.data_path, f"data_{args.data}", f'*_feature.npy'))
    date_strs = [path.split('/')[-1].split('_')[0] for path in feature_paths]
    date_strs.sort()

    if flag == 'train':
        date_list = [date_str for date_str in date_strs if date_str not in test_dates]
    else:
        date_list = [date_str for date_str in date_strs if date_str in test_dates]
        
    print(f"date_list: {date_list}")

    # logger.info(f"train_date_list: {train_date_list}")
    # logger.info(f"test_date_list: {test_date_list}")
    
    Dataset_class = data_dict[args.dataset_class]
    dataset = Dataset_class(data_dir_path=data_dir_path, preprocess_stats_path=args.stats_path, date_list=date_list, rolling_window_stride=args.rolling_window_stride, padding_value=0, history_seq_len=args.history_seq_len, history_label_len=args.history_label_len, future_pre_len=args.future_pre_len, freq_per_second=args.freq_per_second)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True, prefetch_factor=16, pin_memory=True)
    
    return dataset, dataloader

    