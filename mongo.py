import pymongo
import genRandom
import time
import numpy as np
myclient,mydb,col = None,None,None
def mongoConnect():
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')
    mydb = myclient['yymo']
    col = mydb["traincoll"]
    return col
def testtime(qlist, num,k):
    itemlist = genRandom.getitemlist()
    totaltime = 0
    for i in range(0,num):
        d = {}
        d[itemlist[k]] = qlist[i]
        #totaltime = totaltime + json['executionStats']'executionTimeMillis'
        '''
        for w in json:
            print(w)
            '''
        djson = col.find(d).explain()
        totaltime = totaltime + djson['executionStats'][ 'executionTimeMillis']
    return totaltime

def testout(querydata, queryN, sparselist,flied,k):
    timelist = []
    partfilter = {
        flied: {
            "$gte": np.min(querydata),
            "$lte": np.max(querydata),
        }
    }
    #rtime = testtime(querydata, int(sparselist[k]*queryN),k)
    # b树索引
    col.create_index([(flied, pymongo.DESCENDING)],name=flied)
    timelist.append(testtime(querydata, int(sparselist[k]*queryN),k))
    col.drop_index(flied)
    #创建唯一索引
    try:
        col.create_index([(flied, pymongo.DESCENDING)], unique = True,name=flied)
        timelist.append(testtime(querydata, int(sparselist[k] * queryN), k))
        col.drop_index(flied)
    except:
        timelist.append(-1)
        print("not unique")
    col.create_index([(flied, pymongo.DESCENDING)], sparse=True,name=flied)
    timelist.append(testtime(querydata, int(sparselist[k] * queryN), k))
    col.drop_index(flied)

    col.create_index([(flied, pymongo.DESCENDING)], partialFilterExpression=partfilter,name=flied)
    timelist.append(testtime(querydata, int(sparselist[k] * queryN), k))
    col.drop_index(flied)

    # hash索引
    col.create_index([(flied, pymongo.HASHED)], name=flied)
    timelist.append(testtime(querydata, int(sparselist[k] * queryN), k))
    col.drop_index(flied)
    # 创建唯一索引
    try:
        col.create_index([(flied, pymongo.HASHED)], unique=True, name=flied)
        timelist.append(testtime(querydata, int(sparselist[k] * queryN), k))
        col.drop_index(flied)
    except:
        timelist.append(-1)
        print("not unique")
    col.create_index([(flied, pymongo.HASHED)], sparse=True, name=flied)
    timelist.append(testtime(querydata, int(sparselist[k] * queryN), k))
    col.drop_index(flied)

    col.create_index([(flied, pymongo.HASHED)], partialFilterExpression=partfilter, name=flied)
    timelist.append(testtime(querydata, int(sparselist[k] * queryN), k))
    col.drop_index(flied)
    print(k, ",")
    print(timelist)
def testquery(qlist):
    itemlist = genRandom.getitemlist()
    begin = time.perf_counter()
    for i in range(8):
        for j in range(10000):
            print(j)
            if qlist[i][j+10000]:
                partfilter = {
                    itemlist[i]: {
                        "$gte": qlist[i][j],
                        "$lte": qlist[i][j+10000],
                    }
                }
                djson = col.find(partfilter)
            else:
                d = {}
                d[itemlist[i]] = slist[i] + str(qlist[i][j])
                djson = col.find(d)
    end = time.perf_counter()
    totaltime = end-begin
    return totaltime
if __name__=="__main__":
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')
    mydb = myclient['yymo']
    col = mydb["traincoll"]
    slist = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]

    qlist = np.loadtxt("query.txt")
    testquery(qlist)
    '''
    partfilter = {
        'name': {
            "$gte": 'a100',
            "$lte": 'a50000',
        }
    }
    col.create_index([('name', pymongo.HASHED)], partialFilterExpression=partfilter, name='name')

    numrandom = np.loadtxt("numrandom.txt")
    queryNum = 10000
    sparseList = [1, 1, 1, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 1, 0.1, 0.8, 0.2, 0.7, 0.3]
    itemlist = genRandom.getitemlist()
    for i in range(0, 16):
        testout(numrandom[:,i], queryNum, sparseList,itemlist[i], i)
    
    for i in range(0, 16):
        print(i, ":" ,queryNum*sparseList[i])
        print(testtime(numrandom[:,i], int(sparseList[i]*queryNum),i))

mydict = {"name": "Google", "alexa": "1", "url": "https://www.google.com"}

x = mycol.insert_one(mydict)
mycol.create_index()
print(x.inserted_id)
'''