# 生成一个节点个数*样本个数的数组

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy


'''
绘制数据分布图
'''
def pltRes(res):
    n = len(res)
    plt.plot(np.linspace(-5,5,n), res)
    plt.show()
'''
对生成具有一定基数的离散化数据进行标准化，使之和样本数据的个数一致
'''
def normRes(res, exp,seed):
    np.random.seed(seed)
    if exp>0:
        k = np.random.choice(len(res), exp.astype(int), replace=False)
        #k = np.random.randint(0, len(res),exp.astype(int))
        for i in range(0, exp.astype(int)):
            res[k[i]] = res[k[i]] - 1
    elif exp<0:
        exp = (-1)*exp
        k = np.random.choice(len(res), exp.astype(int), replace=False)
        for i in range(0, exp.astype(int)):
            res[k[i]] = res[k[i]] + 1

    print(np.sum(res))
    return res
'''
根据标准正态分布生成样本数目为samnum，基数度为varity，稀疏度为full的整数正态分布
'''
def getNormal(samnum, varity, full, seed=0):
    total = samnum*full
    a = norm.ppf(1/total,0,1) #累计概率密度反函数
    b = norm.ppf(1-1/total,0,1)
    bar = np.linspace(a, b, int(samnum*varity*full))
    a = norm.cdf(bar,0,1)*total #累计概率密度
    b = a[:-1]
    c = np.concatenate([np.zeros(1),b], axis=0)
    d = a-c
    res = np.rint(d)
    exp = np.sum(res) - total
    res = normRes(res, exp, seed)
    return res

def getNormalNum(samnum,full, seed):
    total = samnum*full
    np.random.seed(seed)
    res = np.random.normal(0,1,int(total))

    return res

def getUniformNum(samnum,full, seed):
    total = samnum*full
    np.random.seed(seed)
    res = np.random.uniform(0, samnum, int(total))
    return res

'''
根据均匀生成样本数目为samnum，基数度为varity，稀疏度为full的均匀分布
'''
def getUniform(samnum, varity, full,seed=0):
    total = samnum*full
    a = np.rint(np.ones(int(total * varity)) * (total/(varity*total)))
    exp = np.sum(a) - total
    res = normRes(a, exp, seed)
    return res

'''
将生成的单列数据赋值给数据集
'''
def fillDataset(sam, res, k, list, total, seed):
    np.random.seed(seed)
    idlist = np.random.choice(sam, total, replace=False)
    id = 0
    for i in range(0, list.shape[0]):
        for j in range(0, list[i].astype(int)):
            res[idlist[id]][k] = i
            id = id + 1
    return res
'''
def fillNumData(sam, res,k,list,total,seed):
    np.random.seed(seed)
    idlist = np.random.choice(sam, total, replace=False)
    id = 0
    for i in range(0, list.shape[0]):
        res[idlist[id]][k] = list[i]
        id = id + 1
    return res
    '''
'''
根据样本数量，节点数目，类别列表，基数列表，稀疏列表，随机种子列表生成数据
'''
def genRandomData(sam = 10000, nitem = 0, Lclass = [], Lcard = [], Lsparse = [], Lseed = []):
    res = np.zeros([sam,nitem])
    rdata = []
    for i in range(0, nitem):
        if Lclass[i] == 0:
            # 返回长度为card的数组，count值为数据实际数目
            tmpi = getNormal(sam,Lcard[i], Lsparse[i], Lseed[i])
            #pltRes(tmpi)
            #plt.show()
            res = fillDataset(sam, res, i, tmpi, int(sam*Lsparse[i]), Lseed[i])
        elif Lclass[i] == 1:
            #返回长度为card的数组，count值为数据实际数目
            tmpi = getUniform(sam, Lcard[i], Lsparse[i], Lseed[i])
            res = fillDataset(sam, res, i, tmpi, int(sam*Lsparse[i]), Lseed[i])
            #pltRes(tmpi)
            #plt.show()
        rdata.append(tmpi)
    return res, rdata
'''
根据字典生成子节点列表
'''
def getitemlist():
    dic = {
        "name": 0,
        "age": 0,
        "emailaddress": 0,
        "phone": 0,
        "address": {
            "class1": {
                "street": 0,
                "city": 0,
                "country": 0},
            "class2": {
                "province": 0,
                "zipcode": 0}
        },
        "homepage": 0,
        "profile": {
            "interest": 0,
            "education": 0,
            "gender": 0,
            "netincome": {
                "in": 0,
                "out": 0
            },
            "business": 0
        }
    }
    itemlist = []

    def genSq(d, sq):
        for i in d.keys():
            if isinstance(d[i], dict):
                genSq(d[i], sq+i+'.')
            else:
                itemlist.append(sq+i)
    genSq(dic, '')
    return itemlist
'''
生成一个文档
'''
def genDoc(dlist=[], dic={}):
    nlist = []
    for i in range(16):
        if slist[i]:
            nlist.append(slist[i] + str(dlist[i]))
        else:
            nlist.append(dlist[i])
    lid = []
    lid.append(0)
    def fillDic(d):
        for i in list(d):
            if isinstance(d[i], dict):
                fillDic(d[i])
            else:
                if dlist[lid[0]] == 0:
                    del d[i]
                else:
                    d[i] = nlist[lid[0]]
                lid[0] = lid[0] + 1

    fillDic(dic)
    return dic

'''
qualify doc
'''
def qualifyDoc(dic = {}):
    if dic.__len__() == 0:
        return -1
    for i in list(dic):
        if dic[i]=={}:
            del dic[i]
        elif isinstance(dic[i], dict):
            qualifyDoc(dic[i])

def fillItems(data, sam):
    dic = {
        "name": 0,
        "age": 0,
        "emailaddress":0,
        "phone": 0,
        "address":{
            "class1":{
                "street":0,
                "city":0,
                "country":0},
            "class2":{
                "province":0,
                "zipcode":0}
        },
        "homepage":0,
        "profile":{
            "interest":0,
            "education":0,
            "gender":0,
            "netincome":{
                "in":0,
                "out":0
            },
            "business":0
        }
    }
    for i in range(sam):
        filename = 'D:/dataset/3/person-gen-' + str(i) + '.txt'  # 批量生成的文件名
        f = open(filename,'w')
        ndic = genDoc(list(data[i].astype(int)), copy.deepcopy(dic))
        if qualifyDoc(ndic) != -1:
            print(ndic, file=f)
            f.flush()
            f.close()

def genUnique(res, rdataset, itemlist, num, seed):
    # 随机选取两个字段作为生成查询的基础字段
    np.random.seed(seed)
    s = np.random.randint(0, len(itemlist), 2)
    k1, k2 = itemlist[s[0]],itemlist[s[1]]
    print("unique query flied:" + str(k1) +','+ str(k2))
    mid = np.random.randint(0, num,size=1)[0]
    s1 = np.random.randint(0, len(rdataset[k1]), mid)
    if slist[k1]:
        for i in s1:
            res.append([k1, slist[k1] + str(i)])
    else:
        for i in s1:
            res.append([k1, i])
    s2 = np.random.randint(0,len(rdataset[k2]), num-mid)
    if slist[k2]:
        for i in s2:
            res.append([k2, slist[k2] + str(i)])
    else:
        for i in s2:
            res.append([k2, i])
    return res

def genPartial(res, rdataset, itemlist, num, seed):
    np.random.seed(seed)
    s = np.random.randint(0, len(itemlist), size = 3)
    k1, k2, k3 = itemlist[s[0]],itemlist[s[1]], itemlist[s[2]]
    print("partial query flied:" + str(k1) + ',' + str(k2)+','+str(k3))
    mid1 = np.random.randint(0,num,size=1)[0]
    mid2 = np.random.randint(mid1, num, size = 1)[0]
    # 第一个item取随机区间
    a1 = np.random.randint(0, len(rdataset[k1]),size=1)[0]
    a2 = np.random.randint(a1, len(rdataset[k1]),size=1)[0]
    print(a1,a2)
    s1 = np.random.randint(a1,a2,mid1)
    for i in s1:
        res.append([k1, i])
    # 第二个item取随机区间
    a1 = np.random.randint(0, len(rdataset[k2]),1)[0]
    a2 = np.random.randint(a1, len(rdataset[k2]), size=1)[0]
    print(a1, a2)
    s1 = np.random.randint(a1, a2, mid2-mid1)
    for i in s1:
        res.append([k2, i])

    # 第san个item取随机区间
    a1 = np.random.randint(0, len(rdataset[k3]),1)[0]
    a2 = np.random.randint(a1, len(rdataset[k3]), size=1)[0]
    print(a1,a2)
    s1 = np.random.randint(a1, a2, num-mid2)
    for i in s1:
        res.append([k3, i])
    return res

def genSparse(res, rdataset, itemlist, num, seed):
    np.random.seed(seed)
    s = np.random.randint(0, len(itemlist), size=2)
    k1, k2 = itemlist[s[0]],itemlist[s[1]]
    print("sparse query flied:" + str(k1) + ',' + str(k2))
    mid1 = np.random.randint(0, num, size=1)[0]
    s1 = np.random.randint(0, len(rdataset[k1]),mid1)
    for i in s1:
        res.append([k1, slist[k1] + str(i)])
    s2 = np.random.randint(0, len(rdataset[k2]), num-mid1)
    for i in s2:
        res.append([k2, slist[k2] + str(i)])
    return res

def genRQUery(res, rdataset, itemlist, num, seed):
    np.random.seed(seed)
    s = np.random.randint(0, len(itemlist), size=4)
    k1, k2, k3, k4 = itemlist[s[0]],itemlist[s[1]], itemlist[s[2]],itemlist[s[3]]
    print("random query flied:" + str(k1) + ',' + str(k2))
    print("random query flied:" + str(k2) + ',' + str(k3))
    mid1 = np.random.randint(0, num, size=1)[0]
    mid2 = np.random.randint(mid1, num, size=1)[0]
    mid3 = np.random.randint(mid2, num, size=1)[0]
    s1 = np.random.randint(0, len(rdataset[k1]), mid1)
    for i in s1:
        res.append([k1, slist[k1] + str(i)])
    s1 = np.random.randint(0, len(rdataset[k2]), mid2-mid1)
    for i in s1:
        res.append([k2, slist[k2] + str(i)])
    s1 = np.random.randint(0, len(rdataset[k3]), mid3-mid2)
    for i in s1:
        res.append([k3, slist[k3] + str(i)])
    s1 = np.random.randint(0, len(rdataset[k4]), num-mid3)
    for i in s1:
        res.append([k4, slist[k4] + str(i)])
    return res

def genQuery(num, rdataset, qnum = 1000, itemfeatrue = [], qclass = [], qset=[], seed = 0):
    res = []
    sparsequery = int(qnum * qclass[0]) #稀疏查询个数
    partialquery = int(qnum * qclass[1]) #部分查询个数
    randomquery = int(qnum * qclass[2]) #随机查询个数
    uniquequery = int(qnum * qclass[3])#唯一键查询个数
    #生成唯一键查询
    res = genUnique(res, rdataset, itemfeatrue[0], uniquequery,seed)
    #生成部分查询，首先选择四个部分
    res = genPartial(res, rdataset, itemfeatrue[1], partialquery, seed)
    #生成稀疏查询
    res = genSparse(res, rdataset, itemfeatrue[2], sparsequery, seed)
    #生成随机查询
    res = genRQUery(res,rdataset,itemfeatrue[3], randomquery, seed)

    return res

def writeQuery(querylist,itemset):
    f = open("query.txt", 'a')
    for i in querylist:
        d = {}
        d[itemset[i[0]]] = i[1]
        print(d, file=f)
    f.flush()
    f.close()

if __name__=="__main__":
    '''
    稀疏的字段
    一般字段 
    基数小
    唯一字段 id
    先根据字段选择合适的数据项，根据数据项生成相应的doc，再从doc中提取出查询子文档
    '''
    sampleNum = 500000
    itemNum = 16
    classList = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]
    cardList = [1,1 ,0.8,0.9,0.1,0.5,0.7,0.4,0.9,0.6,1,1 ,0.8,0.9,0.1,0.5,0.7,0.4,0.9,0.6]
    sparseList = [1,0.95, 0.8,0.9,0.8,0.9,0.85,0.15,0.10,0.8,1,0.95, 0.8,0.9,0.8,0.9,0.85,0.15,0.10,0.8]
    seedList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    slist = ["a",0,0,"d","e",0,0,"h","i","g",0,0,"m","n",0,0]
    queryNum = 10000
    queryList = [0.2,0.3,0.2,0.3] #20%稀疏字段查询，30%部分查询，%20随机查询，30%唯一键查询
    itemfeaturelist = [
        [0,10], #唯一键
        [1,2,5,6,10,11,14,15],#数字键
        [7,8],#稀疏键
        [3,4,9,12,13] #一般键
    ]
    itemset = getitemlist()
    data, rdata = genRandomData(sampleNum,itemNum,classList,cardList, sparseList, seedList)
    np.savetxt("2.txt", data.astype(int))

    #data = np.loadtxt("1.txt")
    fillItems(data, sampleNum)
    querydata = genQuery(data, rdata, queryNum,itemfeaturelist, queryList, itemset)
    writeQuery(querydata,itemset)






