# from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
#     MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
# from data_provider.uea import collate_fn
# from torch.utils.data import DataLoader

# data_dict = {
#     'ETTh1': Dataset_ETT_hour,
#     'ETTh2': Dataset_ETT_hour,
#     'ETTm1': Dataset_ETT_minute,
#     'ETTm2': Dataset_ETT_minute,
#     'custom': Dataset_Custom,
#     'm4': Dataset_M4,
#     'PSM': PSMSegLoader,
#     'MSL': MSLSegLoader,
#     'SMAP': SMAPSegLoader,
#     'SMD': SMDSegLoader,
#     'SWAT': SWATSegLoader,
#     'UEA': UEAloader
# }


# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1

#     shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
#     drop_last = False
#     batch_size = args.batch_size
#     freq = args.freq

#     if args.task_name == 'anomaly_detection':
#         drop_last = False
#         data_set = Data(
#             args = args,
#             root_path=args.root_path,
#             win_size=args.seq_len,
#             flag=flag,
#         )
#         print(flag, len(data_set))
#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last)
#         return data_set, data_loader
#     elif args.task_name == 'classification':
#         drop_last = False
#         data_set = Data(
#             args = args,
#             root_path=args.root_path,
#             flag=flag,
#         )

#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last,
#             collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
#         )
#         return data_set, data_loader
#     else:
#         if args.data == 'm4':
#             drop_last = False
#         data_set = Data(
#             args = args,
#             root_path=args.root_path,
#             data_path=args.data_path,
#             flag=flag,
#             size=[args.seq_len, args.label_len, args.pred_len],
#             features=args.features,
#             target=args.target,
#             timeenc=timeenc,
#             freq=freq,
#             seasonal_patterns=args.seasonal_patterns
#         )
#         print(flag, len(data_set))
#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last)
#         return data_set, data_loader

import mycode.dataset as dataset

import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader


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
        
        
    # def load_data(self, data_path_idx):
    #     if self._data_path_idx == data_path_idx:
    #         return 
        
    #     if data_path_idx >= len(self.feature_paths):
    #         raise IndexError("Index out of range")
        
    #     # load data
    #     # _feature_data = np.load(self.feature_paths[data_path_idx]) # [data_len, feature_dim]
    #     _snapshot_data = np.load(self.snapshot_paths[data_path_idx]) # [data_len, snapshot_dim]
    #     _interpolated_mid_prices = np.load(self.interpolated_mid_prices_paths[data_path_idx]) # [data_len, snapshot_dim]
        
    #     # assert _feature_data.shape[0] == _snapshot_data.shape[0], f"feature data shape: {_feature_data.shape}, snapshot data shape: {_snapshot_data.shape}"
        
    #     # # remove timestamp
    #     # self._feature_data = _feature_data[:, 1:]
    #     self._snapshot_data = _snapshot_data[:, 1:]
    #     self._time_stamps = _snapshot_data[:, 0]
    #     self._interpolated_time_stamps = _interpolated_mid_prices[:, 0]
    #     self._interpolated_mid_prices = _interpolated_mid_prices[:, 1]
    #     # convert to float32
    #     # self._feature_data = self._feature_data.astype(np.float32)
    #     self._snapshot_data = self._snapshot_data.astype(np.float32)
    #     # self._time_stamps = self._time_stamps.astype(np.float32)
    #     # self._interpolated_time_stamps = self._interpolated_time_stamps.astype(np.float32)
    #     self._interpolated_mid_prices = self._interpolated_mid_prices.astype(np.float32)
    #     # normalize
    #     # self._feature_data = (self._feature_data - self.feature_mean) / self.feature_std
    #     self._snapshot_data = (self._snapshot_data - self.snapshot_mean) / self.snapshot_std
    #     self._interpolated_mid_prices = (self._interpolated_mid_prices - self.mid_price_mean) / self.mid_price_std
    #     # padding
    #     # self._feature_data = np.pad(self._feature_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='constant', constant_values=self.padding_value)
    #     self._snapshot_data = np.pad(self._snapshot_data, ((self.history_seq_len, self.future_pre_len), (0, 0)), mode='edge')
    #     self._time_stamps = np.pad(self._time_stamps, ((self.history_seq_len, self.future_pre_len),), mode='edge')
    #     self._interpolated_time_stamps = np.pad(self._interpolated_time_stamps, ((self.history_seq_len, self.future_pre_len),), mode='edge')
    #     self._interpolated_mid_prices = np.pad(self._interpolated_mid_prices, ((self.history_seq_len, self.future_pre_len),), mode='edge')
        
    #     # update data path index
    #     self._data_path_idx = data_path_idx
        
    #     # return self._feature_data, self._snapshot_data, self._time_stamps, self._interpolated_time_stamps, self._interpolated_mid_prices
        
        
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
    


data_dict = {
    'FeatureDataset': FeatureDataset,
    'timeSeriesDataset': timeSeriesDataset
}
    

def data_provider(args, flag):
    args.data = 'CMEES'
    args.data_path = './data/'
    args.stats_path = './my_exp/stats/CMEES'
    # args.rolling_window_stride = 100
    if flag == 'val':
        args.rolling_window_stride = 100
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True, prefetch_factor=8)
    
    return dataset, dataloader

    