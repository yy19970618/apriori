# coding=utf-8
import numpy as np
import csv
alphabet = "abcdefghijklmnop0123456789"
alphabet_size = len(alphabet)
import pandas as pd


class Dataset(object):
    def __init__(self):
        self.index_in_epoch = 0
        self.alphabet = alphabet
        self.alphabet_size = alphabet_size

    def dataset_read(self, input,n):
        # 引入embedding矩阵和字典
        embedding_w, embedding_dic = self.onehot_dic_build()

        feature_dim = 30000
        heigh = 7
        weigh = 26
        res = np.zeros([n,feature_dim,heigh,weigh],dtype=np.int8)
        for i in range(0,n):
            for j in range(0, feature_dim):
                vdoc = self.doc_process2(input[i+1][j+1], embedding_dic,embedding_w)
                res[i][j] = vdoc
        #res.tofile("onehot.dat", sep=',')
        return res

    def doc_process1(self, doc, embedding_dic):
        min_len = len(doc)
        doc_vec = np.zeros(26, dtype='int8')
        for j in range(min_len):
            doc_vec[embedding_dic[doc[j]]-1] = 1
        return doc_vec
    def doc_process2(self, doc, embedding_dic, embedding_w):
        min_len = len(doc)
        doc_vec = np.zeros([7, 26], dtype='int8')
        if min_len==1: return doc_vec
        if min_len > 7:
            doc_vec[0] = embedding_w[embedding_dic[doc[0]] - 1]
            for j in range(1, 7):
                doc_vec[j] = embedding_w[embedding_dic['9']-1]
            return doc_vec
        for j in range(0,min_len):
            doc_vec[j] = embedding_w[embedding_dic[doc[j]]-1]
        return doc_vec
    def onehot_dic_build(self):
        # onehot编码
        alphabet = self.alphabet
        embedding_dic = {}
        embedding_w = []
        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='int8')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = np.array(embedding_w, dtype='int8')
        return embedding_w, embedding_dic


if __name__ == "__main__":

    slist = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    train = np.loadtxt("test.csv", delimiter=',')
    a = train.shape[0]
    b = train.shape[1]
    itemnum = 14
    train = train.astype(int)
    strtrain = []
    for i in np.arange(0, a, itemnum):
        for j in range(0, itemnum):
            id = i + j
            strtrain.append([])
            for k in range(0, 30000):
                if train[id][k]:
                    w = slist[j] + str(train[id][k])
                    strtrain[id].append(w)
                else:
                    strtrain[id].append('0')
    del train
    test = pd.DataFrame(data=strtrain)
    test.to_csv('strtest.csv')
    '''
    strtrain = np.loadtxt("strtrain.csv", dtype=np.str,delimiter=',')

    Dataset().dataset_read(strtrain)
    
    test = pd.DataFrame( data=strtrain)
    test.to_csv('strtrain.csv')
    del datasample
    del train
    del test
    Dataset().dataset_read(strtrain)
    '''