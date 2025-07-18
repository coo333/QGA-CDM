#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: Genetic Algorithm (GA) to search discrete Q-matrix (0/1) structure,
integrated with your existing IDCD + train.py + run.py code.

By default, we do:
  1) Flatten a Q-matrix of shape [n_item, n_know] into a bitstring of length n_item * n_know.
  2) Each GA individual is a list of 0/1 representing this Q matrix.
  3) For each individual (candidate Q), we decode it, build an IDCD model,
     run a small-epoch training with train.py, evaluate on valid_data,
     get AUC => as the individual's fitness.
  4) We run GA selection, crossover, mutation, iterate for a few generations.
  5) The final best Q is stored, and we can do a final big training or just test it.

Requires:
    pip install deap

Usage:
    python ga_qsearch.py --train_file train.csv --valid_file valid.csv ... [other args]

You can incorporate or unify this logic with your run.py as needed.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
# Third-party
from deap import base, creator, tools, algorithms

# Your local modules
import torch
import train  # your train.py
import model  # your model.py
import matplotlib.pyplot as plt
##############################################################################
# 1) Parse arguments (similar to your model_parser.py)
##############################################################################

def parse_args():
    # 定义了种群大小ga_pop_size为 200，进化代数--ga_ngen为 10，交叉概率--ga_cxpb为 0.5，变异概率--ga_mutpb为 0.1，
    # ga_small_epoch小训练轮数”：在 GA 评估某个个体的 Q 矩阵时，会进行多少个 epoch 的 mini 训练；
    # --ga_pop_size 20：你可以改大/改小，越大越可能找到好的解，但计算更费时。
    # --ga_ngen 5：迭代的代数，你也可调大(比如 10、20) 让进化充分进行；若时间有限，就用小一点。
    # --ga_small_epoch 2：评估时对每个个体训练 2 epoch；如果你想更精确评估个体质量，
    # 在评估每个 GA 个体时，对 IDCD 模型只做 2 个 epoch 的 Adam 训练，
    # 以迅速得到一个大致的验证集表现 (AUC / ACC / F1 / RMSE)。
    # 可以加大 epoch(比如5epoch)；但评估时间也会大增。
    # --ga_cxpb 0.5, --ga_mutpb 0.1：GA 交叉/变异概率，也可自行修改，
    # 实际要根据实验结果微调。
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='path to train csv')
    parser.add_argument('--valid_file', type=str, help='path to valid csv', default='')
    parser.add_argument('--test_file', type=str, help='path to test csv', default='')
    parser.add_argument('--n_user', type=int, help='Number of users')
    parser.add_argument('--n_item', type=int, help='Number of items')
    parser.add_argument('--n_know', type=int, help='Number of knowledge points')
    parser.add_argument('--user_dim', type=int, default=32)
    parser.add_argument('--item_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00009)
    parser.add_argument('--ga_small_epoch', type=int, default=2, help='small epoch for GA evaluation')
    parser.add_argument('--ga_pop_size', type=int, default=100)
    parser.add_argument('--ga_ngen', type=int, default=10)
    parser.add_argument('--ga_cxpb', type=float, default=0.5)
    parser.add_argument('--ga_mutpb', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    # 选择现有Q矩阵作为一个初始个体
    parser.add_argument('--init_q_file', type=str, default=None, help='Path to initial Q matrix .npy file')
    parser.add_argument('--save_path', type=str, default='./ga_result')


    args = parser.parse_args()
    return args

##############################################################################
# 2) decode / encode Q matrix
##############################################################################

def decode_qmat(individual, n_item, n_know):
    """
    Convert a 1D bitstring => 2D Q matrix of shape [n_item, n_know].
    """
    arr = np.array(individual, dtype=int)
    return arr.reshape((n_item, n_know))

##############################################################################
# 3) Evaluate function => GA -> fitness
##############################################################################

def evaluate_individual(individual,
                        df_train, df_valid,
                        n_user, n_item, n_know,
                        user_dim, item_dim,
                        batch_size, lr,
                        ga_epoch,
                        device='cpu'):
    """
    1) decode individual's Q bitstring => Q matrix
    2) create an IDCD model with that Q (q_aug=None so it's fixed)
    3) run train(...) for a small number of epochs => partial fit
    4) run eval(...) on valid => get e.g. AUC => use as fitness
    """
    Q_candidate = decode_qmat(individual, n_item, n_know)

    # Build IDCD with Q_mat=Q_candidate, disabling q_aug
    # net = model.IDCD(n_user, n_item, n_know,
    #                  user_dim, item_dim,
    #                  Q_mat=Q_candidate,
    #                  device=device,
    #                  q_aug=None,    # We fix Q, no augmentation
    #                  lambda_q=0.0)  # no extra Q regular?
    net = model.IDCD(n_user, n_item, n_know,
                     user_dim, item_dim,
                     Q_mat=Q_candidate,
                     q_aug='single',
                     lambda_q=0.008,
                     device=device)

    # train small epoch
    train.train(net, df_train, df_valid,
                batch_size=batch_size,
                lr=lr,
                n_epoch=ga_epoch)

    # eval on valid
    val_result = train.eval(net, df_valid, batch_size=256)
    # We take ACC as fitness
    fitness = val_result['acc']
    return (fitness,)


##############################################################################
# 4) Main GA routine
##############################################################################

# def run_ga_qsearch(args, df_train, df_valid):
#     """
#     We define a GA to search Q. We'll create population, evaluate, do selection.
#     """
#     n_item, n_know = args.n_item, args.n_know
#
#     # 1) define DEAP structures
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMax)
#
#     toolbox = base.Toolbox()
#
#     # init: random bitstring length = n_item*n_know
#     def init_individual():
#         return [random.randint(0,1) for _ in range(n_item * n_know)]
#
#     toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#     # 2) define evaluate - partial
#     def ga_evaluate(individual):
#         return evaluate_individual(individual,
#                                    df_train, df_valid,
#                                    args.n_user, args.n_item, args.n_know,
#                                    args.user_dim, args.item_dim,
#                                    args.batch_size, args.lr, args.ga_small_epoch,
#                                    device=args.device)
#
#     toolbox.register("evaluate", ga_evaluate)
#     # selection => tournament
#     toolbox.register("select", tools.selTournament, tournsize=3)
#     # crossover => two-point
#     toolbox.register("mate", tools.cxTwoPoint)
#     # mutation => flipbit
#     toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
#
#     # 3) create population
#     pop_size = args.ga_pop_size
#     ngen = args.ga_ngen
#     pop = toolbox.population(n=pop_size)
#
#     # optional Hall of Fame
#     hof = tools.HallOfFame(1)
#
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)
#
#     # 4) run GA
#     final_pop, logbook = algorithms.eaSimple(pop, toolbox,
#                                              cxpb=args.ga_cxpb,
#                                              mutpb=args.ga_mutpb,
#                                              ngen=ngen,
#                                              stats=stats,
#                                              halloffame=hof,
#                                              verbose=True)
#
#     records = [dict(r) for r in logbook]  # each r is a record for generation
#     df_log = pd.DataFrame(records)  # "gen", "nevals", "avg", "std", "min", "max"
#     if not os.path.exists(args.save_path):
#         os.makedirs(args.save_path)
#     csv_path = os.path.join(args.save_path, "ga_log.csv")
#     df_log.to_csv(csv_path, index=False)
#     print(f"GA log saved to: {csv_path}. Columns: {df_log.columns.tolist()}")
#     best_ind = hof[0]
#     best_fitness = best_ind.fitness.values[0]
#     best_q_mat = decode_qmat(best_ind, n_item, n_know)
#
#     print(f"[GA] Best Q found with fitness={best_fitness:.4f}")
#     return best_q_mat

# 选择现有Q矩阵作为一个初始个体
def run_ga_qsearch(args, df_train, df_valid):
    """
    We define a GA to search Q. We'll create population, evaluate, do selection.
    """
    n_item, n_know = args.n_item, args.n_know

    # 修复点 1: 先定义 DEAP 的 Fitness 和 Individual 类
    # --------------------------------------------------
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    # --------------------------------------------------

    # 修复点 2: 加载现有 Q 矩阵（如果存在）
    existing_ind = None
    if args.init_q_file:
        existing_q = np.load(args.init_q_file)
        if existing_q.shape != (n_item, n_know):
            raise ValueError(f"Initial Q matrix shape {existing_q.shape} does not match ({n_item}, {n_know})")
        existing_flat = existing_q.flatten().tolist()
        existing_ind = creator.Individual(existing_flat)  # 现在 creator.Individual 已定义
        existing_ind.fitness.values = (0.0,)   # 初始适应度置空

    # 配置 DEAP 工具箱
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: [random.randint(0, 1) for _ in range(n_item * n_know)])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 定义评估函数
    def ga_evaluate(individual):
        return evaluate_individual(individual,
                                   df_train, df_valid,
                                   args.n_user, args.n_item, args.n_know,
                                   args.user_dim, args.item_dim,
                                   args.batch_size, args.lr, args.ga_small_epoch,
                                   device=args.device)

    toolbox.register("evaluate", ga_evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)

    # 创建初始种群
    pop_size = args.ga_pop_size
    if existing_ind:
        pop = [existing_ind]
        pop += toolbox.population(n=pop_size - 1)
    else:
        pop = toolbox.population(n=pop_size)

    # 运行遗传算法
    ngen = args.ga_ngen
    # hof = tools.HallOfFame(1)
    hof = tools.HallOfFame(5)  # 修改为保存前 5 个最优个体
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    final_pop, logbook = algorithms.eaSimple(pop, toolbox,
                                             cxpb=args.ga_cxpb,
                                             mutpb=args.ga_mutpb,
                                             ngen=ngen,
                                             stats=stats,
                                             halloffame=hof,
                                             verbose=True)

    records = [dict(r) for r in logbook]  # each r is a record for generation
    df_log = pd.DataFrame(records)  # "gen", "nevals", "avg", "std", "min", "max"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    csv_path = os.path.join(args.save_path, "ga_log.csv")
    df_log.to_csv(csv_path, index=False)
    print(f"GA log saved to: {csv_path}. Columns: {df_log.columns.tolist()}")
    # best_ind = hof[0]
    # best_fitness = best_ind.fitness.values[0]
    # best_q_mat = decode_qmat(best_ind, n_item, n_know)
    #
    # print(f"[GA] Best Q found with fitness={best_fitness:.4f}")
    # return best_q_mat
    # 保存前 10 个最优的 Q 矩阵
    for i, ind in enumerate(hof):
        fitness = ind.fitness.values[0]
        q_mat = decode_qmat(ind, n_item, n_know)
        q_mat_path = os.path.join(args.save_path, f'best_q_matrix_top{i + 1}.npy')
        np.save(q_mat_path, q_mat)
        print(f"[GA] Top {i + 1} Q found with fitness={fitness:.4f}, saved to {q_mat_path}")

    return hof  # 可以选择返回 HallOfFame 对象，方便后续使用

def main():
    args = parse_args()

    # load data
    df_train = pd.read_csv(args.train_file)
    df_valid = pd.read_csv(args.valid_file)

    # run GA search
    # best_q_mat = run_ga_qsearch(args, df_train, df_valid)
    hof = run_ga_qsearch(args, df_train, df_valid)
    # done => optionally do final big training
    best_ind = hof[0]
    best_q_mat = decode_qmat(best_ind, args.n_item, args.n_know)
    print("Now let's do final training with best Q:")
    # net = model.IDCD(args.n_user, args.n_item, args.n_know,
    #                  args.user_dim, args.item_dim,
    #                  Q_mat=best_q_mat,
    #                  device=args.device,
    #                  q_aug=None,
    #                  lambda_q=0.0)
    net = model.IDCD(
        args.n_user, args.n_item, args.n_know,
        args.user_dim, args.item_dim,
        Q_mat=best_q_mat,
        device=args.device,
        q_aug='single',
        lambda_q=0.008
    )

    # train longer e.g. 10 epochs
    train.train(net, df_train, df_valid,
                batch_size=args.batch_size,
                lr=args.lr,
                n_epoch=30,
                save_path=args.save_path)

    # if you have test set:
    if args.test_file:
        df_test = pd.read_csv(args.test_file)
        test_result = train.eval(net, df_test, batch_size=256)
        print("Test result after final big train =>", test_result)
    else:
        print("No test file provided. Done.")

    # save best Q if desired
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    np.save(os.path.join(args.save_path, 'math2best_q_matrix.npy'), best_q_mat)
    return args.save_path

def plot_ga_log(log_csv_path):
        df = pd.read_csv(log_csv_path)
        # typical columns: gen, nevals, avg, std, min, max
        plt.figure()
        plt.plot(df["gen"], df["max"], label="Max Fitness")
        plt.plot(df["gen"], df["avg"], label="Avg Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("GA Convergence")
        plt.legend()
        plt.grid(True)
        plt.show()
if __name__ == "__main__":
    save_path = main()
    log_csv_path = os.path.join(save_path, "ga_log.csv")
    plot_ga_log(log_csv_path)
