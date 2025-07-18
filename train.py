import gc
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from model import IDCD

torch.set_default_tensor_type(torch.FloatTensor)


class IDCDataset(Dataset):
    def __init__(self, df_log: pd.DataFrame, n_user: int, n_item: int, Q_mat=None):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        pbar = tqdm(total=df_log.shape[0], desc='Loading data')
        for i, row in df_log.iterrows():
            self.log_mat[int(row['user_id']), int(row['item_id'])] \
                = (row['score'] - 0.5) * 2
            pbar.update(1)
        pbar.close()

    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        return torch.Tensor(self.log_mat[user_id, :]), \
            torch.Tensor(self.log_mat[:, item_id]), \
            torch.LongTensor([user_id]), \
            torch.LongTensor([item_id]), \
            torch.FloatTensor([self.score[index]]) \

    def __len__(self):

        return self.user_id.shape[0]


def compute_loss(model: IDCD, pred: torch.Tensor, score: torch.Tensor):
    """
    计算总损失：BCE + Q 的正则 (可选)
    """
    # 1) 基础BCE损失
    bce_loss = F.binary_cross_entropy(pred, score)

    # 2) 如果没有开启 Q 增强 (q_aug=None), 直接返回
    if not model.q_aug:
        return bce_loss

    # 3) 否则，根据q_aug的类型对可学习Q做正则
    reg_loss = 0.0

    if model.q_aug == 'single':
        # model.q_neural: [n_item, n_know]
        # sigmoided 后保证 [0,1]
        q_aug = torch.sigmoid(model.q_neural)  # (n_item, n_know)

        # 惩罚先验=0位置与q_aug差异：mask出先验=0
        # mask_zero = (model.Q_mat == 0).float()
        # diff = (q_aug - model.Q_mat) * mask_zero
        # reg_loss = model.lambda_q * diff.abs().sum()

        # 对全部位置做差异惩罚：
        diff = q_aug - model.Q_mat
        reg_loss = model.lambda_q * diff.abs().sum()

    elif model.q_aug == 'mf':
        # A.weight: [n_item, dim], B.weight: [n_know, dim]
        # => q_mf_full: [n_item, n_know]
        q_mf_full = torch.sigmoid(model.A.weight @ model.B.weight.T)

        # 同理可先 mask 先验=0 if仅想学那些
        # mask_zero = (model.Q_mat == 0).float()
        # diff = (q_mf_full - model.Q_mat) * mask_zero
        # reg_loss = model.lambda_q * diff.abs().sum()

        diff = q_mf_full - model.Q_mat
        reg_loss = model.lambda_q * diff.abs().sum()

    total_loss = bce_loss + reg_loss
    return total_loss


def train(model: IDCD, train_data: pd.DataFrame, valid_data: pd.DataFrame, \
          batch_size, lr, n_epoch, save_path: str = None):
    model.train()
    device = model.device
    dataset = IDCDataset(train_data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print(model.Theta_buf.is_leaf, model.Psi_buf.is_leaf)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    result_per_epoch = []
    for epoch in range(n_epoch):
        result_epoch = {}
        pbar = tqdm(total=len(dataloader), desc='Epoch %d' % epoch)
        score_all = []
        pred_all = []
        Theta_old = model.get_Theta_buf().numpy().copy()
        for i, (user_log, item_log, user_id, item_id, score) in enumerate(dataloader):
            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)
            pred = model(user_log, item_log, user_id, item_id)
            # alpha_example = model.knowledge_attn.last_alpha  # 取出forward存的alpha
            # # 只打印第一个 batch 的前几个 alpha
            # if i == 0:
            #     print(f"[Epoch {epoch} Batch {i}] alpha example:", alpha_example[0])
            loss = compute_loss(model, pred, score)
            score_all += score.detach().cpu().numpy().reshape(-1, ).tolist()
            pred_all += pred.detach().cpu().numpy().reshape(-1, ).tolist()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if model.q_aug == 'single':
                with torch.no_grad():
                    model.q_neural.data.clamp_(0.0, 1.0)
            pbar.update(1)
        pbar.close()

        model.eval()

        # Update examinee traits
        n_user = dataset.log_mat.shape[0]  # user数
        for i in range(math.ceil(n_user / batch_size)):
            idx = np.arange(i * batch_size, min(n_user, (i + 1) * batch_size))
            user_log_batch = torch.Tensor(dataset.log_mat[idx, :]).to(device)
            theta_new = model.diagnose_theta(user_log_batch).detach()
            model.update_Theta_buf(theta_new, torch.LongTensor(idx))

        # -- 更新 question features (Psi_buf)
        n_item = dataset.log_mat.shape[1]
        for i in range(math.ceil(n_item / batch_size)):
            idx = np.arange(i * batch_size, min(n_item, (i + 1) * batch_size))
            item_log_batch = torch.Tensor(dataset.log_mat[:, idx].T).to(device)
            psi_new = model.diagnose_psi(item_log_batch).detach()
            model.update_Psi_buf(psi_new, torch.LongTensor(idx))

        model.train()

        # =========================
        # 训练集上一些指标
        # =========================
        score_all = np.array(score_all)
        pred_all = np.array(pred_all)
        acc = accuracy_score(score_all, (pred_all > 0.5))
        Theta_new = model.get_Theta_buf().numpy().copy()
        Theta_norm = np.sqrt(np.sum(np.abs(Theta_new - Theta_old)))
        try:
            auc_value = roc_auc_score(score_all, pred_all)
        except ValueError:
            auc_value = 0.0

        print(f'epoch = {epoch}, theta_norm = {Theta_norm:.6f}, acc = {acc:.6f}, auc = {auc_value:.6f}')

        result_epoch['Theta_old_head'] = Theta_old[:5, :5]
        result_epoch['Theta_new_head'] = Theta_new[:5, :5]
        result_epoch['Theta_norm'] = Theta_norm
        result_epoch['train_eval'] = {'acc': acc, 'auc': auc_value}

        # =========================
        # 验证集
        # =========================
        if valid_data is not None:
            result_epoch['valid_eval'] = eval(model, valid_data, batch_size=16)

        result_per_epoch.append(result_epoch)

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")

    return result_per_epoch


def get_eval_result(s_true, s_pred, s_pred_label):
    acc = accuracy_score(s_true, s_pred_label)
    f1 = f1_score(s_true, s_pred_label)
    rmse = np.sqrt(mean_squared_error(s_true, s_pred))
    try:
        auc_value = roc_auc_score(s_true, s_pred)
        # s_pred 是模型输出的概率；s_true 是 0/1
    except ValueError:
        auc_value = 0.0

    print(f'acc = {acc:.6f} f1 = {f1:.6f} rmse = {rmse:.6f} auc = {auc_value:.6f}')

    return {'acc': acc, 'f1': f1, 'rmse': rmse, 'auc': auc_value}


def eval(model: IDCD, data: pd.DataFrame, batch_size):
    """
    使用 forward_using_buf 评估
    """
    model.eval()
    device = model.device
    dataset = IDCDataset(data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    y_pred = []
    for i, (user_log, item_log, user_id, item_id, score) in enumerate(dataloader):
        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id = user_id.to(device)
        item_id = item_id.to(device)

        # forward_using_buf 会在内部也使用 get_Q_enhanced
        pred_1_batch = model.forward_using_buf(user_id, item_id).detach().cpu().tolist()
        y_pred += pred_1_batch

    y_pred = np.array(y_pred)
    y_true = data['score'].values.astype(int)
    y_plab = (y_pred > 0.5).astype(int)
    eval_result = get_eval_result(y_true, y_pred, y_plab)
    return eval_result