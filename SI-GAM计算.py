import numpy as np

def fuzzyfun(alpha,q_m,prob_num):
    '''
    # 目前只能实现每次计算一道题的eta值
    # :param alpha: array, (stu_m, knowledge_k)，学生的知识点掌握度
    # :param mu: array, (2^(knowledge_k),)，单个试题的模糊测度
    # :param mu: array, (2^(knowledge_k),)，单个试题的模糊测度
    # :return:prob_num,表示试题的数量
    # '''

    stu_num=alpha.shape[0]
    eta=np.zeros((stu_num, prob_num))
    for i in range(0, prob_num):
        a = alpha * q_m[i]
        idx = np.argwhere(np.all(a[..., :] == 0, axis=0))
        alpha1 = np.delete(a, idx, axis=1)#处理后的alpha，选择本题对应的知识点计算
        mu = np.loadtxt("./data/D2/mu/mu_ms{}.csv".format(i + 1), delimiter=",", skiprows=0)#每道题目对应一个模糊测度问津
        mu =mu[:, int(sum(q_m[i])):]#适用于q矩阵几个知识点就几个k
        alpha_sorted = - np.sort(- alpha1)  # 降序排序
        inds = np.argsort(- alpha1) # 降序排序，排序后的的下标
        inds_2 = 2 ** (inds)#001表示x1
        mu_rows = np.cumsum(inds_2, axis=1)  # (stu_m, knowledge_k)
        mu_m = mu[mu_rows].reshape(stu_num,-1)  # (stu_m, knowledge_k)
        eta[:, i] = np.max(np.minimum(mu_m, alpha_sorted), axis=1)


    return eta   # (stu_m,)




