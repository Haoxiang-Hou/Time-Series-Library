from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim

import os
import sys
import time
import tqdm
import logging
import json

import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

from mycode import ic_compute

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self): 
        class MaskedMSELoss(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.args = args

                self.mask_select_list = []
                for _next_time in [0.5, 1, 3, 5, 10, 20, 60, 120]:
                    _mask_idx = int(_next_time * self.args.freq_per_second) - 1
                    if _mask_idx < self.args.pred_len:
                        self.mask_select_list.append(_mask_idx)
                    else:
                        self.mask_select_list.append(self.args.pred_len - 1)
                        break
                print(f"mask_select_list: {self.mask_select_list}")
                
                self.mask = torch.zeros((self.args.pred_len, self.args.target_size), dtype=torch.bool)
                for idx in self.mask_select_list:
                    self.mask[idx, :] = True
                
                self.MSELoss = nn.MSELoss()
                
            def forward(self, input, target):
                # input/target: [batch, seq, dim]
                # 只在 idx_list 位置计算 loss
                _mask = self.mask.unsqueeze(0).expand(input.shape[0], -1, -1) # [batch, pred_len, target_size]
                input_masked = input[_mask]
                target_masked = target[_mask]
                return self.MSELoss(input_masked, target_masked)
            
            
        if self.args.loss_function == 'MSELoss':       
            criterion = nn.MSELoss()
        elif self.args.loss_function == 'MaskedMSELoss':
            criterion = MaskedMSELoss(self.args)
        else:
            raise ValueError(f"Loss function {self.args.loss} not supported")
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        pre_labels = []
        interpolat_price_labels = []
        ref_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, labels) in tqdm.tqdm(enumerate(vali_loader), desc='Validation', total=len(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else -self.args.target_size
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                _batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = _batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                
                if self.args.target_size==1: # output is mid price seq
                    current_last_midprice = batch_x[:, -1, -1].unsqueeze(-1).detach().cpu() # shape: [batch_size, 1]
                    
                    pre_future_midprices = torch.cat([current_last_midprice, pred.squeeze(-1)], dim=1) # shape: [batch_size, pred_len+1]
                    true = torch.cat([current_last_midprice, true.squeeze(-1)], dim=1) # shape: [batch_size, pred_len+1]
                    
                    pre_label = ic_compute.compute_label_from_pre_future_midprice_with_index(current_last_midprice, pre_future_midprices, raw_time_intervals=[0.5, 1, 3, 5, 10, 20, 60, 120], time_index_factor=self.args.freq_per_second, transform_func=vali_data.inverse_transform)
                    
                    interpolat_price_label = ic_compute.compute_label_from_pre_future_midprice_with_index(current_last_midprice, true, raw_time_intervals=[0.5, 1, 3, 5, 10, 20, 60, 120], time_index_factor=self.args.freq_per_second, transform_func=vali_data.inverse_transform)
                    
                    # assert pre_label.shape == (batch_x.shape[0], 8), f"pre_label shape: {pre_label.shape}, batch_x shape: {batch_x.shape}"
                    # assert interpolat_price_label.shape == (batch_x.shape[0], 8), f"interpolat_price_label shape: {interpolat_price_label.shape}, batch_x shape: {batch_x.shape}"
                elif self.args.target_size==8: # output is labels
                    pre_label = pred.squeeze(1).detach().cpu() # shape: [batch_size, self.args.target_size
                    assert pre_label.shape == (batch_x.shape[0], self.args.target_size), f"pre_label shape: {pre_label.shape}, batch_x shape: {batch_x.shape}"
                    interpolat_price_label = _batch_y.squeeze(1).detach().cpu() # shape: [batch_size, self.args.target_size]
                    assert interpolat_price_label.shape == (batch_x.shape[0], self.args.target_size), f"interpolat_price_label shape: {interpolat_price_label.shape}, batch_x shape: {batch_x.shape}"
                else:
                    raise ValueError(f"target_size {self.args.target_size} not supported, only 1 and 8 are supported")
                
                pre_labels.append(pre_label)
                interpolat_price_labels.append(interpolat_price_label)
                ref_labels.append(labels)

        total_loss = np.average(total_loss)
        
        pre_labels = np.concatenate(pre_labels, axis=0)
        interpolat_price_labels = np.concatenate(interpolat_price_labels, axis=0) # [num_windows, 8]
        ref_labels = np.concatenate(ref_labels, axis=0) # [num_windows, 8]
        
        ic_pre_price = ic_compute.get_ic(pre_labels, interpolat_price_labels)
        ic_pre_ref = ic_compute.get_ic(pre_labels, ref_labels)
        ic_price_ref = ic_compute.get_ic(interpolat_price_labels, ref_labels)
        
        self.model.train()
        return total_loss, ic_pre_price, ic_pre_ref, ic_price_ref

    def train(self, setting, logger):
        logger.info(self.model)
        params_num = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {params_num:,}")
        train_params_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {train_params_num:,}")
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        # if os.path.exists(path):
        #     # stop training if os.path.dirname(model_save_path) exists
        #     print(f"{path} exists, stop training")
        #     sys.exit(0)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, labels) in tqdm.tqdm(enumerate(train_loader), desc='Training', total=len(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else -self.args.target_size
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else -self.args.target_size
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logging.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            logger.info("Epoch: {} train cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            vali_loss, ic_pre_price, ic_pre_ref, ic_price_ref = self.vali(vali_data, vali_loader, criterion)
            logger.info("Epoch: {} train + vali cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                # epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | ic_pre_price: {json.dumps(ic_pre_price)}, ic_pre_ref: {json.dumps(ic_pre_ref)}, ic_price_ref: {json.dumps(ic_price_ref)}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
