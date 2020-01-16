# 生成一个节点个数*样本个数的数组

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy
import mongo
from onehot import *
'''
绘制数据分布图
'''
def pltRes(res):
    n = len(res)
    plt.scatter(np.linspace(-5,5,n), res)
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
    slist = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "g", "k", "l", "m", "n", "o", "p"]

    nlist = []
    for i in range(14):
        if slist[i]:
            nlist.append(slist[i] + str(dlist[i]))
        else:
            nlist.append(int(dlist[i]))
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
        #"emailaddress":0,
        #"phone": 0,
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
    col = mongo.mongoConnect()
    for i in range(sam):
        #filename = 'D:/dataset/5/person-gen-' + str(i) + '.txt'  # 批量生成的文件名
        #f = open(filename,'w')
        print(i)
        ndic = genDoc(list(data[i]), copy.deepcopy(dic))
        if qualifyDoc(ndic) != -1:
            col.insert(ndic)
            '''
            print(ndic, file=f)
            f.flush()
            f.close()
            '''

def extractData(rdataset, num, itemN):
    res = np.zeros([num,itemN])
    for i in range(0, len(rdataset)):
        id = 0
        for j in range(0,len(rdataset[i])):
            for k in range(int(rdataset[i][j])):
                res[id][i] = j
                id = id + 1
    #np.savetxt("rdata.txt", res)
    return res
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

def genQuery(rdataset, qnum = 1000, itemfeatrue = [], qclass = [], seed = 0):
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
    d = {}
    d[itemset[querylist[0][0]]] = querylist[0][1]
    print(d, file=f)
    for i in range(1,len(querylist)):
        d = {}
        d[itemset[querylist[i][0]]] = querylist[i][1]
        if querylist[i][0] != querylist[i-1][0]:
            print(i)
            f.flush()
            f.close()
            f = open((itemset[querylist[i][0]]+ ".txt"), 'a')
        print(d, file=f)
    f.flush()
    f.close()


'''
进行分层抽样，数字字段按照数字大小分段，字符串字段按照稀疏度进行分层
'''
def getNumSamples(rdataset,itemclass=[],sampleN=0, sparselist = [], seed = 0):
    samres = np.zeros([sampleN, len(itemclass)])
    np.random.seed(seed)
    for i in range(0, len(itemclass)):
        #字符串字段,按照稀疏度分层抽样
        if itemclass[i] :
            n = sampleN*sparselist[i]
            sam_id = np.random.randint(0, len(rdataset)*sparselist[i], int(n))
            for j in range(0,int(n)):
                samres[j][i] = rdataset[sam_id[j]][i]
        else:
        #数字字段按照数字大小分层
            n = sampleN*sparselist[i]
            rlist = rdataset[:, i]
            tlist = []
            bar = (np.max(rlist)-np.min(rlist))/100
            lastid,id = 0,0
            for j in range(1, 101):
                maxk = np.min(rlist) + j*bar
                while rdataset[id][i] < maxk:
                    id = id + 1
                    if id == 1000000:
                        id = 999999
                        break

                barn = np.ceil(((id-lastid)/(len(rdataset)*sparselist[i]))*n)
                sam_id = np.random.randint(lastid, id,int(barn))
                lastid = id
                for item in sam_id:
                    tlist.append(rdataset[item][i])
            #pltRes(tlist)
            norid = np.random.randint(0, tlist.__len__(), int(n))
            for k in range(0,int(n)):
                samres[k][i] =tlist[norid[k]]
        #pltRes(np.sort(samres[:,i]))
    #np.savetxt("samres1.txt", samres)
    return samres

def genRandomNumQuery(Num,rdataset, sparselist,querynum, flag,seed):
    np.random.seed(seed)
    res1 = np.zeros([querynum,len(sparselist)])
    res2 = np.zeros([querynum, len(sparselist)])
    for i in range(0, len(sparselist)):
        s = np.random.randint(0, sparselist[i]*Num,int(querynum))
        for j in range(0, len(s)):
            res1[j][i] = rdataset[s[j]][i]
        partnum = int(querynum * flag)
        partlist = np.random.randint(0, 20, partnum)
        for j in range(0, partnum):
            res2[j][i] = rdataset[s[j]][i] + partlist[j]
    #np.savetxt("numrandom1.txt", res)
    #a = np.flipud(np.sort(res1,axis=0))
    #b = np.flipud(np.sort(res2,axis=0))
    a = np.flipud(np.sort(res1, axis=0))[:, seed]
    b = np.flipud(np.sort(res2, axis=0))[:, seed]
    return np.concatenate((a, b)), np.concatenate((res1[:,seed],res2[:,seed]))
    #return np.concatenate((a,b)),res1[:,seed],res2[:,seed]

def genpartNumQuery(Num,rdataset, sparselist,querynum,flag, seed):
    np.random.seed(seed)
    res1 = np.zeros([querynum,len(sparselist)])
    res2 = np.zeros([querynum, len(sparselist)])
    for i in range(0, len(sparselist)):
        tnum = querynum
        a = np.random.randint(0, sparselist[i]*Num-2000, 1)[0]
        b= np.random.randint(a, sparselist[i] * Num, 1)[0]
        s = np.random.randint(a, b,int(tnum) )
        for j in range(0, len(s)):
            res1[j][i] = rdataset[s[j]][i]
        partnum = int(querynum*flag)
        partlist = np.random.randint(0,20, partnum)
        for j in range(0,len(partlist)):
            res2[j][i] = rdataset[s[j]][i] + partlist[j]
    #a = np.flipud(np.sort(res1, axis=0))
    #b = np.flipud(np.sort(res2, axis=0))
    a = np.flipud(np.sort(res1, axis=0))[:,seed]
    b = np.flipud(np.sort(res2, axis=0))[:,seed]
    return np.concatenate((a,b)),np.concatenate((res1[:,seed],res2[:,seed]))
    #np.savetxt("numpart_r.txt", partres)
#随机查询：单值和range查询各256组
def gentraindata1(epoch, rdata,flag,res):
    for i in range(0,epoch):
        # res = np.concatenate((res,out))
        b = genRandomNumQuery(totalNum, rdata, sparseList, 10000, flag,i)
        b = b.transpose()
        res = np.concatenate((res,b))
    return res
#部分查询：单值和range查询各256组
def gentraindata2(epoch, rdata,res):

    flag = 0
    for i in range(0, epoch):
        # res = np.concatenate((res,out))
        b = genpartNumQuery(totalNum, rdata, sparseList, 10000, flag, i)
        b = b.transpose()
        res = np.concatenate((res, b))
    flag = 1
    for i in range(0, epoch):
        # res = np.concatenate((res,out))
        b = genpartNumQuery(totalNum, rdata, sparseList, 10000, flag, i)
        b = b.transpose()
        res = np.concatenate((res, b))
    return res
#混合了单值和range查询，取占比为0.2,0.8 一共16*2*16组
def gentraindata3(epoch,rdata,res):

    for i in range(0,8):
        a = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0.9, i)
        a = a.transpose()
        res = np.concatenate((res,a))
    for i in range(0,8):
        c = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0.1, i)
        c = c.transpose()
        res = np.concatenate((res,c))
    for i in range(0,8):
        a = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0.8, i)
        a = a.transpose()
        res = np.concatenate((res,a))
    for i in range(0,8):
        c = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0.2, i)
        c = c.transpose()
        res = np.concatenate((res,c))
    return res

#16组大查询，即查询结果占总数据的百分之40以上
def genlargerange(Num,rdataset, sparselist,querynum,seed):
    np.random.seed(seed)
    res1 = np.zeros([querynum,len(sparselist)])
    res2 = np.zeros([querynum, len(sparselist)])
    for i in range(0, len(sparselist)):
        index = sparselist[i]*Num
        s = np.random.randint(0, index*0.5,int(querynum))
        for j in range(0, len(s)):
            res1[j][i] = rdataset[s[j]][i]
        partlist = np.random.randint(index*0.4, index*0.5, int(querynum))
        for j in range(0, len(s)):
            res2[j][i] = rdataset[s[j]][i] + partlist[j]
    #np.savetxt("numrandom1.txt", res)
    #a = np.flipud(np.sort(res1,axis=0))
    #b = np.flipud(np.sort(res2,axis=0))
    a = np.flipud(np.sort(res1, axis=0))[:, seed]
    b = np.flipud(np.sort(res2, axis=0))[:, seed]
    return np.concatenate((a, b)), np.concatenate((res1[:, seed], res2[:, seed]), axis=0)
    #return np.concatenate((a,b)), res1[:,seed],res2[:,seed]
def genlargerange2(Num,rdataset, sparselist,querynum,seed):
    np.random.seed(seed)
    res1 = np.zeros([querynum,len(sparselist)])
    res2 = np.zeros([querynum, len(sparselist)])
    for i in range(0, len(sparselist)):
        index = sparselist[i]*Num
        s = np.random.randint(0, index*0.4,int(querynum))
        for j in range(0, len(s)):
            res1[j][i] = rdataset[s[j]][i]
        partlist = np.random.randint(index*0.5, index*0.6, int(querynum))
        for j in range(0, len(s)):
            res2[j][i] = rdataset[s[j]][i] + partlist[j]
    #np.savetxt("numrandom1.txt", res)
    #a = np.flipud(np.sort(res1,axis=0))
    #b = np.flipud(np.sort(res2,axis=0))
    a = np.flipud(np.sort(res1, axis=0))[:, seed]
    b = np.flipud(np.sort(res2, axis=0))[:, seed]
    return np.concatenate((a, b)), np.concatenate((res1[:, seed], res2[:, seed]), axis=0)
    #return np.concatenate((a,b)),res1[:,seed],res2[:,seed]

def gentraindata4(epoch, rdata,res):
    for i in range(0, epoch):
        a = genlargerange(totalNum, rdata, sparseList, 10000, i)
        a = a.transpose()
        res = np.concatenate((res, a))
    for i in range(0, epoch):
        a = genlargerange2(totalNum, rdata, sparseList, 10000, i)
        a = a.transpose()
        res = np.concatenate((res, a))
    return res
if __name__=="__main__":

    #np.savetxt("out.txt",out)

    totalNum = 1000000
    itemNum = 16
    classList = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
    cardList = [1,0.1,1,0.2,0.8,0.3,0.7,0.4,0.6,1,0.1,1,0.2,0.8,0.3,0.7]
    sparseList = [1,1,0.1,0.8,0.2,0.7,0.3,0.6,0.4,1,1,0.1,0.8,0.2,0.7,0.3]
    seedList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #slist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    slist = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]
    train = np.zeros([8,20000])
    query = np.zeros([8,20000])
    data, rdata = genRandomData(totalNum, itemNum, classList, cardList, sparseList, seedList)
    #np.savetxt("data.txt", data.astype(int))
    rdata = extractData(rdata, totalNum, itemNum)
    #前面八个字段抽取20000个，后面八个字段抽取100个，生成原始工作负载1
    a,b = genRandomNumQuery(totalNum,rdata,sparseList,10000,0,0)
    train[0],query[0] = a, b
    a,b = genRandomNumQuery(totalNum,rdata,sparseList,10000,1,1)
    train[1], query[1] = a, b
    a, b = genpartNumQuery(totalNum, rdata, sparseList, 10000, 0, 2)
    train[2], query[2] = a, b
    a, b = genpartNumQuery(totalNum, rdata, sparseList, 10000, 1, 3)
    train[3], query[3] = a, b
    a, b = genpartNumQuery(totalNum, rdata, sparseList, 10000, 0.9, 4)
    train[4],query[4] = a, b
    a, b = genpartNumQuery(totalNum, rdata, sparseList, 10000, 0.1, 5)
    train[5], query[5] = a, b
    a,b = genlargerange(totalNum, rdata, sparseList,10000,6)
    train[6], query[6] = a, b
    a, b = genlargerange(totalNum, rdata, sparseList, 10000, 7)
    train[7], query[7] = a, b
    datasample = getNumSamples(rdata, slist, 10000, sparseList, 0)
    datasample = np.sort(datasample, axis=0)
    datasample = datasample.transpose()[0:8,:]
    test = np.concatenate((datasample,train),axis=1)
    #genstrnumpy(test, 8, "test1.csv")
    np.savetxt("query.txt",query)
    # 对原数据进行数字查询生成，可以根据不同的seed生成n多组数据
    train = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0, 0)
    train = train.transpose()
    for i in range(1, 16):
        # res = np.concatenate((res,out))
        b = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0, i)
        b = b.transpose()
        train = np.concatenate((train, b))

    train = gentraindata1(16, rdata, 1, train)
    train = gentraindata2(16, rdata, train)
    train = gentraindata3(16, rdata, train)
    train = gentraindata4(16, rdata, train)
    #np.savetxt("train.csv", train, delimiter=',')

    #生成数据样本
    datasample = getNumSamples(rdata, slist, 10000, sparseList, 0)
    datasample = np.sort(datasample, axis=0)
    datasample = datasample.transpose()
    # 拼接抽样数据和查询数据
    #input = np.loadtxt("train.csv", delimiter=',')
    #data = np.loadtxt("datatrain.csv", delimiter=',')
    bar = 2048/16
    out = datasample
    for i in range(1, int(bar)):
        out = np.concatenate((out,datasample))
    train = np.concatenate((out, train),axis=1)
    res = genstrnumpy(train, itemNum, "train.csv")
    #np.savetxt("numtrain.csv", train, delimiter=',')
    '''
    out = np.zeros([2048,4])
    # 查询是大规模的，不需要建立索引
    for i in range(0, 1024 + 512):
        out[i][0] = 1
    #字段是否稀疏
    dim1 = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    for i in range(0,1024+512):
        id = i%16
        out[i][1] = dim1[id]
    #查询是否部分,部分查询有32epoch，从512开始
    for i in range(512, 1024):
        out[i][2] = 1
    #查询是b树or hash
    for i in range(256,512):
        out[i][3] = 1
    for i in range(768,1024):
        out[i][3] = 1
    for i in range(1024, 1024+128):
        out[i][3] = 1
        out[i+256][3] = 1

    np.savetxt("out.csv", out, delimiter=',')
    
    
    根据数据生成文档，再将这些文档存储到数据库中去
    data = np.loadtxt("2.txt")
    fillItems(data, totalNum)
    '''






