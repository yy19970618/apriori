from  genRandom import *

if __name__=="__main__":
    totalNum = 1000000
    itemNum = 14
    classList = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    cardList = [1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    sparseList = [1,0.8,0.2,0.7,0.3,0.6,0.4,1,0.8,0.2,0.7,0.3,0.6,0.4]
    seedList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14]
    slist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #slist = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    # queryNum = 5000*16
    out = np.zeros([7*14, 4])
    # 字段是否稀疏
    dim1 = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    for i in range(0, 6*14):
        id = i % itemNum
        out[i][0] = dim1[id]
    # 查询是否部分,部分查询有32epoch，从512开始
    for i in range(2*14, 4*14):
        out[i][1] = 1
    # 查询是b树or hash
    for i in range(14, 28):
        out[i][2] = 1
    for i in range(14+28, 28+28):
        out[i][2] = 1
    for i in range(14*5, 14*6):
        out[i][2] = 1
    # 查询是大规模的，不需要建立索引
    for i in range(0, 14*6):
        out[i][3] = 1
    np.savetxt("testout.csv", out, delimiter=',')
    '''
    data, rdata = genRandomData(totalNum, itemNum, classList, cardList, sparseList, seedList)
    rdata = extractData(rdata, totalNum, itemNum)
    datasample = getNumSamples(rdata, slist, 10000, sparseList, 0)
    datasample = np.flipud(np.sort(datasample, axis=0))
    datasample = datasample.transpose()
    query = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0, 0)
    query = query.transpose()
    b = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 1, 0)
    b = b.transpose()
    query = np.concatenate((query, b))
    b = genpartNumQuery(totalNum, rdata, sparseList, 10000, 0, 0)
    b = b.transpose()
    query = np.concatenate((query, b))
    b = genpartNumQuery(totalNum, rdata, sparseList, 10000, 1, 0)
    b = b.transpose()
    query = np.concatenate((query, b))
    b = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0.9, 0)
    b = b.transpose()
    query = np.concatenate((query, b))
    b = genRandomNumQuery(totalNum, rdata, sparseList, 10000, 0.1, 0)
    b = b.transpose()
    query = np.concatenate((query, b))
    b = genlargerange(totalNum, rdata, sparseList, 10000, 0)
    b = b.transpose()
    query = np.concatenate((query, b))

    n = 7
    out = datasample
    for i in range(1, n):
        out = np.concatenate((out, datasample))
    test = np.concatenate((out, query), axis=1)
    np.savetxt("test.csv", test, delimiter=',')
    '''