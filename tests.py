# xiaorongD
# import torch
# import model
# import train
# import pandas as pd
# import numpy as np
#
# # 加载数据
# # math1
# # df_test = pd.read_csv("data/math1_test_0.8_0.2.csv")
# # # 构造 IDCD，注意传入与训练时一致的超参数
# # q_mat = np.load("data/math1_Q_matrix.npy")
# # net_checkpoint = model.IDCD(n_user=4209,
# #                             n_item=15,
# #                             n_know=11,
# #                             user_dim=32,
# #                             item_dim=32,
# #                             Q_mat=q_mat,      # 与训练时保持一致
# #                             device=torch.device('cpu'),
# #                             # q_aug='single', # 如果训练时用single
# #                             q_aug=None,    # 或者''，保持与训练时一致
# #                             dim=32,
# #                             )
# # df_test = pd.read_csv("data/math2_test_0.8_0.2.csv")
# # # 构造 IDCD，注意传入与训练时一致的超参数
# # q_mat = np.load("data/math2_Q_matrix.npy")  # 如果你保存了的话
# # net_checkpoint = model.IDCD(n_user=3911,
# #                             n_item=16,
# #                             n_know=16,
# #                             user_dim=32,
# #                             item_dim=32,
# #                             Q_mat=q_mat,      # 与训练时保持一致
# #                             device=torch.device('cpu'),
# #                             # q_aug='single', # 如果训练时用single
# #                             # q_aug=None,    # 或者''，保持与训练时一致
# #                             dim=32,
# #                             )
# # # junyi
# df_test = pd.read_csv("data/junyi_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("data/junyi_Q_matrix.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=10000,
#                             n_item=734,
#                             n_know=734,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             # q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
#
# # k =7
# # checkpoint_path = f"./math1result/xiaorongD/checkpoint_epoch_{k}.pth"
# # net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# # k =3
# # checkpoint_path = f"./math2result/xiaorongD/checkpoint_epoch_{k}.pth"
# # net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# k =17
# checkpoint_path = f"./junyiresult/xiaorongD/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# #
# # # 测试
# # test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# # print(f"[Checkpoint {k}] Test result =>", test_result)
#
# # 如果上述代码有键对匹配的问题，则启用下面代码忽略键对。
# checkpoint = torch.load(checkpoint_path)
#
# # 获取当前模型的 state_dict
# model_state_dict = net_checkpoint.state_dict()
#
# # 移除不匹配的键
# unexpected_keys = []
# mismatched_keys = []
# for key in checkpoint.keys():
#     if key not in model_state_dict:
#         unexpected_keys.append(key)
#     elif checkpoint[key].shape != model_state_dict[key].shape:
#         mismatched_keys.append(key)
#
# # 移除意外的键
# for key in unexpected_keys:
#     del checkpoint[key]
#
# # 移除形状不匹配的键
# for key in mismatched_keys:
#     del checkpoint[key]
#
# # 加载处理后的 state_dict
# net_checkpoint.load_state_dict(checkpoint, strict=False)
#
# # 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)


# xiaorongGD
# import torch
# import model
# import train
# import pandas as pd
# import numpy as np
#
# # 加载数据
# # math1
# # df_test = pd.read_csv("data/math1_test_0.8_0.2.csv")
# # # 构造 IDCD，注意传入与训练时一致的超参数
# # q_mat = np.load("math1result/zhiyouG/best_q_matrix_top1.npy")
# # net_checkpoint = model.IDCD(n_user=4209,
# #                             n_item=15,
# #                             n_know=11,
# #                             user_dim=32,
# #                             item_dim=32,
# #                             Q_mat=q_mat,      # 与训练时保持一致
# #                             device=torch.device('cpu'),
# #                             # q_aug='single', # 如果训练时用single
# #                             q_aug=None,    # 或者''，保持与训练时一致
# #                             dim=32,
# #                             )
# # df_test = pd.read_csv("data/math2_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# # q_mat = np.load("math2result/zhiyouG/best_q_matrix_top1.npy")  # 如果你保存了的话
# # net_checkpoint = model.IDCD(n_user=3911,
# #                             n_item=16,
# #                             n_know=16,
# #                             user_dim=32,
# #                             item_dim=32,
# #                             Q_mat=q_mat,      # 与训练时保持一致
# #                             device=torch.device('cpu'),
# #                             # q_aug='single', # 如果训练时用single
# #                             # q_aug=None,    # 或者''，保持与训练时一致
# #                             dim=32,
# #                             )
# # # junyi
# df_test = pd.read_csv("data/junyi_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("junyiresult/zhiyouG/best_q_matrix_top1.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=10000,
#                             n_item=734,
#                             n_know=734,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             # q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# k =3
# checkpoint_path = f"./math1result/xiaorongGD/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))

# k =5
# checkpoint_path = f"./math2result/xiaorongGD/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# k =22
# checkpoint_path = f"./junyiresult/xiaorongGD/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))
#
# # 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)

# 如果上述代码有键对匹配的问题，则启用下面代码忽略键对。
# checkpoint = torch.load(checkpoint_path)
#
# # 获取当前模型的 state_dict
# model_state_dict = net_checkpoint.state_dict()
#
# # 移除不匹配的键
# unexpected_keys = []
# mismatched_keys = []
# for key in checkpoint.keys():
#     if key not in model_state_dict:
#         unexpected_keys.append(key)
#     elif checkpoint[key].shape != model_state_dict[key].shape:
#         mismatched_keys.append(key)
#
# # 移除意外的键
# for key in unexpected_keys:
#     del checkpoint[key]
#
# # 移除形状不匹配的键
# for key in mismatched_keys:
#     del checkpoint[key]
#
# # 加载处理后的 state_dict
# net_checkpoint.load_state_dict(checkpoint, strict=False)
#
# # 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)


# xiaorongQ
import torch
import model
import train
import pandas as pd
import numpy as np

# 加载数据
# math1
# df_test = pd.read_csv("data/math1_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("data/math1_Q_matrix.npy")
# net_checkpoint = model.IDCD(n_user=4209,
#                             n_item=15,
#                             n_know=11,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# df_test = pd.read_csv("data/math2_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("data/math2_Q_matrix.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=3911,
#                             n_item=16,
#                             n_know=16,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# junyi
# df_test = pd.read_csv("data/junyi_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("data/junyi_Q_matrix.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=10000,
#                             n_item=734,
#                             n_know=734,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# # k =2
# # checkpoint_path = f"./math1result/xiaorongQ/checkpoint_epoch_{k}.pth"
# # net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# #
# # k =5
# # checkpoint_path = f"./math2result/xiaorongQ/checkpoint_epoch_{k}.pth"
# # net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# k =0
# checkpoint_path = f"./junyiresult/xiaorongQ/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))
#
# # 测试
# # test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# # print(f"[Checkpoint {k}] Test result =>", test_result)
#
# # 如果上述代码有键对匹配的问题，则启用下面代码忽略键对。
# checkpoint = torch.load(checkpoint_path)
#
# # 获取当前模型的 state_dict
# model_state_dict = net_checkpoint.state_dict()
#
# # 移除不匹配的键
# unexpected_keys = []
# mismatched_keys = []
# for key in checkpoint.keys():
#     if key not in model_state_dict:
#         unexpected_keys.append(key)
#     elif checkpoint[key].shape != model_state_dict[key].shape:
#         mismatched_keys.append(key)
#
# # 移除意外的键
# for key in unexpected_keys:
#     del checkpoint[key]
#
# # 移除形状不匹配的键
# for key in mismatched_keys:
#     del checkpoint[key]
#
# # 加载处理后的 state_dict
# net_checkpoint.load_state_dict(checkpoint, strict=False)
#
# # 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)


# xiaorongGQ
import torch
import model
import train
import pandas as pd
import numpy as np

# 加载数据
# math1
# df_test = pd.read_csv("data/math1_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("math1result/zhiyouG/best_q_matrix_top1.npy")
# net_checkpoint = model.IDCD(n_user=4209,
#                             n_item=15,
#                             n_know=11,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# df_test = pd.read_csv("data/math2_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("math2result/zhiyouG/best_q_matrix_top1.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=3911,
#                             n_item=16,
#                             n_know=16,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# junyi
# df_test = pd.read_csv("data/junyi_test_0.8_0.2.csv")
# # # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("junyiresult/zhiyouG/best_q_matrix_top1.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=10000,
#                             n_item=734,
#                             n_know=734,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# k =2
# checkpoint_path = f"./math1result/xiaorongGQ/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))
#
# k =2
# checkpoint_path = f"./math2result/xiaorongGQ/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# k =8
# checkpoint_path = f"./junyiresult/xiaorongGQ/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))

# 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)

# 如果上述代码有键对匹配的问题，则启用下面代码忽略键对。
# checkpoint = torch.load(checkpoint_path)
#
# # 获取当前模型的 state_dict
# model_state_dict = net_checkpoint.state_dict()
#
# # 移除不匹配的键
# unexpected_keys = []
# mismatched_keys = []
# for key in checkpoint.keys():
#     if key not in model_state_dict:
#         unexpected_keys.append(key)
#     elif checkpoint[key].shape != model_state_dict[key].shape:
#         mismatched_keys.append(key)
#
# # 移除意外的键
# for key in unexpected_keys:
#     del checkpoint[key]
#
# # 移除形状不匹配的键
# for key in mismatched_keys:
#     del checkpoint[key]
#
# # 加载处理后的 state_dict
# net_checkpoint.load_state_dict(checkpoint, strict=False)
#
# # 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)



# xiaorongG
# import torch
# import model
# import train
# import pandas as pd
# import numpy as np
# #
# # # 加载数据
# # # math1
# # df_test = pd.read_csv("data/math1_test_0.8_0.2.csv")
# # # 构造 IDCD，注意传入与训练时一致的超参数
# # q_mat = np.load("math1result/zhiyouG/best_q_matrix_top1.npy")
# # net_checkpoint = model.IDCD(n_user=4209,
# #                             n_item=15,
# #                             n_know=11,
# #                             user_dim=32,
# #                             item_dim=32,
# #                             Q_mat=q_mat,      # 与训练时保持一致
# #                             device=torch.device('cpu'),
# #                             # q_aug='single', # 如果训练时用single
# #                             q_aug=None,    # 或者''，保持与训练时一致
# #                             dim=32,
# #                             )
# df_test = pd.read_csv("data/math2_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("math2result/zhiyouG/best_q_matrix_top1.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=3911,
#                             n_item=16,
#                             n_know=16,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             # q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# # junyi
# df_test = pd.read_csv("data/junyi_test_0.8_0.2.csv")
# # 构造 IDCD，注意传入与训练时一致的超参数
# q_mat = np.load("junyiresult/zhiyouG/best_q_matrix_top1.npy")  # 如果你保存了的话
# net_checkpoint = model.IDCD(n_user=10000,
#                             n_item=734,
#                             n_know=734,
#                             user_dim=32,
#                             item_dim=32,
#                             Q_mat=q_mat,      # 与训练时保持一致
#                             device=torch.device('cpu'),
#                             # q_aug='single', # 如果训练时用single
#                             # q_aug=None,    # 或者''，保持与训练时一致
#                             dim=32,
#                             )
# k =2
# checkpoint_path = f"./math1result/xiaorongG/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))

# k =3
# checkpoint_path = f"./math2result/xiaorongG/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))
# k =1
# checkpoint_path = f"./junyiresult/xiaorongG/checkpoint_epoch_{k}.pth"
# net_checkpoint.load_state_dict(torch.load(checkpoint_path))

# 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)

# 如果上述代码有键对匹配的问题，则启用下面代码忽略键对。
# checkpoint = torch.load(checkpoint_path)
#
# # 获取当前模型的 state_dict
# model_state_dict = net_checkpoint.state_dict()
#
# # 移除不匹配的键
# unexpected_keys = []
# mismatched_keys = []
# for key in checkpoint.keys():
#     if key not in model_state_dict:
#         unexpected_keys.append(key)
#     elif checkpoint[key].shape != model_state_dict[key].shape:
#         mismatched_keys.append(key)
#
# # 移除意外的键
# for key in unexpected_keys:
#     del checkpoint[key]
#
# # 移除形状不匹配的键
# for key in mismatched_keys:
#     del checkpoint[key]
#
# # 加载处理后的 state_dict
# net_checkpoint.load_state_dict(checkpoint, strict=False)
#
# # 测试
# test_result = train.eval(net_checkpoint, df_test, batch_size=32)
# print(f"[Checkpoint {k}] Test result =>", test_result)
