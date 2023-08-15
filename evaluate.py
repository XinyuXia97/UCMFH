import numpy as np
import scipy.spatial
from tqdm import tqdm

def fx_calc_map_label(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()  # [batch, batch]
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def fx_calc_recall(self, image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = (dist.argsort() + 1).T  # [batch, batch]
    label_matrix = np.zeros((label.shape[0], label.shape[0]))
    # for i in range(label.shape[0]):
    #     index = np.where(label[i] == label)[0]
    #     label_matrix[i][index] = 1\
    for i in range(label.shape[0]):
        for j in range(label.shape[0]):
            if label[i] == label[j]:
                label_matrix[i, j] = 1
    # _dict = {
    #     'ord':ord,
    #     'label_matrix': label_matrix
    # }
    # sio.savemat(str(self.data_ratio)+'pascal.mat',_dict)
    return ord, label_matrix

    # ranks = np.zeros(image.shape[0])
    # for i in range(image.shape[0]):
    #     q_label = label[i]
    #     r_labels = label[ord[i]]
    #     ranks[i] = np.where(r_labels == q_label)[0][0]
    # print(ranks)

    # # R@K
    # for i in range(image.shape[0]):
    #     q_label = label[i]
    #     r_labels = label[ord[i]]

    #     ranks[i] = np.where(r_labels == q_label)[0][0]
    #     # print(np.where(r_labels == q_label))

    # r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    # r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    # r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Prec@K
    # for K in [1, 2, 4, 8, 16]:
    #     prec_at_k = calc_precision_at_K(ord, label, K)
    #     print("P@{} : {:.3f}".format(k, 100 * prec_at_k))

    # return r1, r5, r10


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, int(tsum))  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
