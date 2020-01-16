from  genRandom import *

if __name__=="__main__":

    #data = np.loadtxt("data.txt")

    totalNum = 1000000
    itemNum = 14
    classList = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    cardList = [1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    sparseList = [1,0.8,0.2,0.7,0.3,0.6,0.4,1,0.8,0.2,0.7,0.3,0.6,0.4]
    seedList = [16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    slist = ["a","b","c","d","e","f","g",0,0,0,0,0,0,0]
    #slist = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]

    data, rdata = genRandomData(totalNum, itemNum, classList, cardList, sparseList, seedList)
    fillItems(data, 1000000)
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


    test = np.concatenate((datasample, query), axis=1)
    np.savetxt("test.csv", test, delimiter=',')
