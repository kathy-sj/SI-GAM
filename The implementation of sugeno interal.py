import numpy as np


def fuzzyfun(alpha,mu):
    '''
    目前只能实现每次计算一道题的eta值
    :param alpha: array, (stu_m, knowledge_k)，学生的知识点掌握度
    :param mu: array, (2^(knowledge_k),)，单个试题的模糊测度
    :return:
    '''
    alpha = np.array(alpha)
    mu = np.array(mu)
    stu_m=alpha.shape[0]
    knowledge_k = alpha.shape[1]
    alpha_sorted = - np.sort(- alpha)   # 降序排序
    inds = np.argsort(- alpha)  # 降序排序，排序后的的下标
    inds_2 = 2 ** (knowledge_k - 1 - inds)   # 先计算保存，减少循环里的重复计算
    mu_rows = np.cumsum(inds_2, axis=1)   # (stu_m, knowledge_k)
    mu_m = mu[mu_rows] # (stu_m, knowledge_k)
    eta = np.max(np.minimum(mu_m, alpha_sorted), axis=1)
    # print(eta)


    return eta   # (stu_m,)

# if __name__ == '__main__':
#     alpha=[[0.9,0.2,0.3],[0.4,0.1,0.5]]
#     mu=[[0,0,0,0],[0,0,1,0.2],[0,1,0.4],[0,1,1,0.6],[1,0,0,0.7],[1,0,1,1],[1,1,0,1],[1,1,1,1]]
#     # mu=np.loadtxt("../Sugeno-CDM/data/mu_strategy—construct.csv", delimiter=",", skiprows=0)
#     mu=np.array(mu).reshape(-1)
#     alpha=np.array(alpha).reshape(-1)
#     a=fuzzyfun(alpha, mu)
#     print(a)


