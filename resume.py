#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
import numpy as np

import train  # 引入上面写的 train.py
import model  # 你的 IDCD 定义
# 假设你还有 parse_args 或手写参数

if __name__ == "__main__":
    # 1) 准备好数据 DataFrame
    df_train = pd.read_csv("data/math1_train_0.8_0.2.csv")
    df_valid = pd.read_csv("data/math1_valid_0.8_0.2.csv")

    # 2) 构造与原先相同的 IDCD 超参
    n_user = 4209
    n_item = 15
    n_know = 11
    user_dim = 32
    item_dim = 32
    device = torch.device('cpu')
    q_aug = 'single'
    lambda_q = 0.008
    dim = 32

    # 假设你训练使用的 Q_mat (可能是best_q_mat)
    best_q_mat = np.load("./result/GA_Qsearch_fixed/best_q_matrix.npy")

    net = model.IDCD(
        n_user, n_item, n_know,
        user_dim, item_dim,
        Q_mat=best_q_mat,
        device=device,
        q_aug=q_aug,
        lambda_q=lambda_q,
        dim=dim
    )

    # 3) 加载 checkpoint
    #    例如, checkpoint_epoch_9.pth
    resume_epoch = 9
    save_path = "./result/GA_Qsearch_fixed"
    checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{resume_epoch}.pth")

    checkpoint_data = torch.load(checkpoint_path, map_location=device)

    # 提取 epoch, model_state, optimizer_state
    loaded_epoch = checkpoint_data['epoch']
    model_state = checkpoint_data['model_state_dict']
    optimizer_state = checkpoint_data['optimizer_state_dict']

    print(f"Loaded checkpoint from epoch={loaded_epoch}")

    # 把 state_dict 加载进 net
    net.load_state_dict(model_state)
    net.train()  # 设置训练模式

    # 4) 继续训练: epoch=10..15
    #    这里给 train.train() 传入 start_epoch=loaded_epoch+1
    #    并把 optimizer_state 也带给它
    total_epoch = 30  # 假设你想训练到 0..15
    train.train(
        net,
        df_train,
        df_valid,
        batch_size=32,
        lr=0.00009,
        n_epoch=total_epoch,
        save_path=save_path,
        start_epoch=loaded_epoch+1,   # => 10
        optimizer_state=optimizer_state
    )

    # 5) 训练完可再 eval
    df_test = pd.read_csv("data/math1_test_0.8_0.2.csv")
    test_result = train.eval(net, df_test, batch_size=32)
    print("Test result after resuming =>", test_result)
