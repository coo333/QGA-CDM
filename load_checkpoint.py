import torch
import model
import train
import pandas as pd
import numpy as np

# 1) 加载测试集
df_test = pd.read_csv("data/math2_test_0.8_0.2.csv")

# 2) 与训练时相同的超参
n_user = 3911
n_item = 16
n_know = 16
user_dim = 32
item_dim = 32
device = torch.device('cpu')

# 3) 构造与训练时相同结构的模型
#    (关键: Q_mat同之前最优Q. 如果你是GA最优Q, 要用best_q_mat.npy或手动加载)
best_q_mat = np.load("./math2result/Top20QGA_Qsearch_fixed/best_q_matrix_top1.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user, n_item, n_know,
#                             user_dim, item_dim,
#                             Q_mat=best_q_mat,
#                             device=device,
#                             q_aug=None,
#                             lambda_q=0.0)
net_checkpoint = model.IDCD(n_user, n_item, n_know,
                            user_dim, item_dim,
                            Q_mat=best_q_mat,
                            device=device,
                            q_aug='single',
                            lambda_q=0.008)
# net_checkpoint = model.IDCD(n_user, n_item, n_know,
#                             user_dim, item_dim,
#                             Q_mat=best_q_mat,
#                             device=device,
#                             )

# 4) 选定要加载的epoch
k = 4
checkpoint_path = f"./math2result/ceshi4/checkpoint_epoch_{k}.pth"

# 5) 加载 state dict
net_checkpoint.load_state_dict(torch.load(checkpoint_path, map_location=device))
net_checkpoint.eval()

# 6) 用 train.eval(...) 在测试集测性能
test_result_k = train.eval(net_checkpoint, df_test, batch_size=32)
print(f"[Checkpoint epoch {k}] => Test result:", test_result_k)
