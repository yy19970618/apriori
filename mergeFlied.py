import copy
import genRandom as gr
import numpy as np
def genset(d):
    s = set()
def getRandomSamples(data = [], sam = 0,sampleNum = 0, seed = 0):
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

    np.random.seed(seed)
    s = np.random.randint(0,sam, sampleNum)
    for i in s:
        filename = 'D:/dataset/sample/3/person-gen-' + str(i) + '.txt'  # 批量生成的文件名
        f = open(filename, 'w')
        ndic = gr.genDoc(list(data[i].astype(int)), copy.deepcopy(dic))
        if gr.qualifyDoc(ndic) != -1:
            print(ndic, file=f)
            f.flush()
            f.close()

def getPartial(filename):
    qlist = {}
    for line in open(filename,'r'):
        d = eval(line)
        for i in d.keys():
            qlist[d[i]] = qlist.get(d[i],0) + 1
    res = sorted(qlist.items(),key=lambda a:a[1], reverse=True)
    print(res[0])

if __name__=="__main__":

    #data = np.loadtxt("2.txt")
    #getRandomSamples(data, 500000, 10000, 0)
    getPartial("profile.netincome.in.txt")