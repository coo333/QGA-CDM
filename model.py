from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
class DotAttention(nn.Module):
        def __init__(self):
            super(DotAttention, self).__init__()
            self.last_alpha = None  # 用于临时存储注意力

        def forward(self, theta_, psi_):
            score = theta_ * psi_  # [batch_size, n_know]
            alpha = torch.softmax(score, dim=1)  # [batch_size, n_know]

            # 存一下 alpha，便于外部访问
            self.last_alpha = alpha.detach()  # detach
            return alpha
class IDCD(nn.Module):
    def __init__(self, n_user: int, n_item: int, n_know: int,user_dim: int, item_dim: int,Q_mat: np.array = None, \
        monotonicity_assumption: bool = True,device = torch.device('cpu'), q_aug: str = None, dim: int = 32, lambda_q: float = 0.0):
        super(IDCD,self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_know = n_know
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.itf = self.ncd_func
        self.device = device
        self.q_aug = q_aug
        self.lambda_q = lambda_q
        self.knowledge_attn = DotAttention()
        self.guess = nn.Embedding(n_item, 1)
        self.slip = nn.Embedding(n_item, 1)

        self.Q_mat = torch.Tensor(Q_mat) if Q_mat is not None else torch.ones((n_item, n_know))
        # 困难度特征
        self.K_diff_mat = nn.Parameter(torch.zeros((n_know, user_dim)),requires_grad=False).to(device)
        self.K_diff_mat.requires_grad = True
        # 学习者特征的问题潜在参数
        self.latent_Zm_emb = nn.Embedding(self.n_user, self.n_know)
        # 问题参数的学习者潜在特征
        self.latent_Zd_emb = nn.Embedding(self.n_item, self.n_know)
        # 问题区分度
        self.e_discrimination = nn.Embedding(self.n_item, 1)

        self.Q_mat = self.Q_mat.to(device)
        if self.q_aug == 'single':
            # 直接把 q_neural 当做 nn.Parameter
            self.q_neural = nn.Parameter(torch.zeros((n_item, n_know),device=device), requires_grad=True)
            nn.init.xavier_normal_(self.q_neural)


        elif self.q_aug == 'mf':
            # 矩阵分解形式
            self.A = nn.Embedding(n_item, dim)  # item_dim
            self.B = nn.Embedding(n_know, dim)  # knowledge_dim
            # 初始化
            nn.init.xavier_normal_(self.A.weight)
            nn.init.xavier_normal_(self.B.weight)

        # Buffer of examinee traits
        self.Theta_buf = nn.Parameter(torch.zeros((n_user, n_know)), requires_grad=False).to(device)
        # Buffer of question feature traits
        self.Psi_buf = nn.Parameter(torch.zeros((n_item, n_know)), requires_grad=False).to(device)
        f_linear = nn.Linear if monotonicity_assumption is False else PosLinear

        self.f_nn = nn.Sequential(
            OrderedDict(
                [
                    ('f_layer_1', f_linear(n_item, 256)),
                    ('f_activate_1', nn.Sigmoid()),
                    ('f_layer_2', f_linear(256, n_know)),
                    ('f_activate_2', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.g_nn = nn.Sequential(
            OrderedDict(
                [
                    ('g_layer_1', nn.Linear(n_user, 512)),
                    ('g_activate_1', nn.Sigmoid()),
                    ('g_layer_2', nn.Linear(512, 256)),
                    ('g_activate_2', nn.Sigmoid()),
                    ('g_layer_3', nn.Linear(256, n_know)),
                    ('g_activate_3', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.theta_agg_mat = f_linear(n_know, user_dim).to(device)
        self.psi_agg_mat = nn.Linear(n_know, item_dim).to(device)

        self.ncd = nn.Sequential(
            OrderedDict([
                ('pred_layer_1', nn.Linear(user_dim, 64)),
                ('pred_activate_1', nn.Sigmoid()),
                ('pred_dropout_1', nn.Dropout(p=0.5)),
                ('pred_layer_2', nn.Linear(64, 32)),
                ('pred_activate_2', nn.Sigmoid()),
                ('pred_dropout_2', nn.Dropout(p=0.5)),
                ('pred_layer_3', nn.Linear(32, 1)),
                ('pred_activate_3', nn.Sigmoid()),

            ])
        ).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def ncd_func(self, theta, psi):
        assert self.user_dim == self.item_dim
        y_pred = self.ncd(theta - psi)
        return y_pred

    def diagnose_theta(self, user_log: torch.Tensor):
        theta = self.f_nn(user_log)
        return theta

    def diagnose_psi(self, item_log: torch.Tensor):
        psi = self.g_nn(item_log)
        return psi

    def diagnose_theta_psi(self, user_log: torch.Tensor, item_log: torch.Tensor):
        theta = self.diagnose_theta(user_log)
        psi = self.diagnose_psi(item_log)
        return theta, psi

    def update_Theta_buf(self, theta_new, user_id):
        self.Theta_buf[user_id] = theta_new

    def update_Psi_buf(self, psi_new, item_id):
        self.Psi_buf[item_id] = psi_new

    def predict_response(self, theta, psi, Q_batch):

        # theta_agg = self.theta_agg_mat(theta * Q_batch)
        # psi_agg = self.psi_agg_mat(psi * Q_batch)
        # output = self.itf(theta_agg, psi_agg)
        # return output
        # 1) 先做对 Q_batch 的 Mask
        theta_ = theta * Q_batch  # [batch_size, n_know]
        psi_ = psi * Q_batch  # [batch_size, n_know]

        # 2) 通过注意力计算 alpha
        alpha = self.knowledge_attn(theta_, psi_)  # [batch_size, n_know]

        # 3) 用 alpha 来加权
        theta_att = alpha * theta_  # [batch_size, n_know]
        psi_att = alpha * psi_

        # 4) 再用原先的线性映射
        theta_agg = self.theta_agg_mat(theta_att)
        psi_agg = self.psi_agg_mat(psi_att)
        # 5) 计算最终预测
        output = self.itf(theta_agg, psi_agg)
        return output

    def forward(self, user_log: torch.Tensor, item_log: torch.Tensor, user_id: torch.LongTensor,
                item_id: torch.LongTensor):
        # latent_zm = self.nonlinear_func(self.latent_Zm_emb(user_id))
        # latend_zd = self.nonlinear_func(self.latent_Zd_emb(item_id))
        # e_disc = torch.sigmoid(self.e_discrimination(item_id))*10
        # identity = torch.eye(self.n_know).to(self.device)
        theta, psi = self.diagnose_theta_psi(user_log, item_log)
        Q_enhanced = self.get_Q_enhanced(item_id)   #下面实现
        output = self.predict_response(theta, psi, Q_enhanced)
        return output

    def get_Q_enhanced(self, item_id: torch.LongTensor):
        """根据 q_aug 模式返回 (batch_size, n_know) 的可学习 Q 矩阵切片"""
        if self.q_aug == 'single':
            # 先把 q_neural 映射到 [0,1] 防止出现负数
            qn_item = torch.sigmoid(self.q_neural[item_id])
            # 与先验 Q 结合：
            Q_enhanced = (1 - self.Q_mat[item_id]) * qn_item + self.Q_mat[item_id]
            # 或者直接返回可学习Q
            # Q_enhanced = qn_item

        elif self.q_aug == 'mf':
            # 做矩阵分解
            # A.weight: [n_item, dim], B.weight: [n_know, dim]
            # (n_item, n_know)
            q_mf_full = torch.sigmoid(self.A.weight @ self.B.weight.T)
            qn_item = q_mf_full[item_id]
            # 与先验 Q 结合
            Q_enhanced = (1 - self.Q_mat[item_id]) * qn_item + self.Q_mat[item_id]
            # Q_enhanced = qn_item

        else:
            # 不做增强，直接用先验 Q
            Q_enhanced = self.Q_mat[item_id]

        return Q_enhanced.squeeze(dim=1)
    def forward_using_buf(self, user_id: torch.LongTensor, item_id: torch.LongTensor):
        theta = self.Theta_buf[user_id].squeeze(dim=1)
        psi = self.Psi_buf[item_id].squeeze(dim=1)
        Q_enhanced = self.get_Q_enhanced(item_id)
        output = self.predict_response(theta, psi, Q_enhanced)
        return output

    def get_Theta_buf(self):
        result = self.Theta_buf.detach().cpu()
        return result

    def get_Psi_buf(self):
        result = self.Psi_buf.detach().cpu()
        return result
