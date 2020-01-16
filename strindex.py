import tensorflow.python as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from onehot import *
def convolution():

    inn = layers.Input(shape=(sequence_length,alpha_len,embedding_dimension,1))
    cnns = []
    for i,size in enumerate(filter_sizes):
        conv = layers.Conv3D(filters=2, kernel_size=([size,alpha_len,embedding_dimension]),
                            strides=[size,1,1], padding='valid', activation='relu')(inn)
        #if i%2:
        pool_size =int(conv.shape[1]/100)
        pool = layers.MaxPool3D(pool_size=([pool_size,1,1]), padding='valid')(conv)
        #pool = MaxMin(pool_size)(conv)

        cnns.append(pool)
    outt = layers.concatenate(cnns)

    model = keras.Model(inputs=inn, outputs=outt,name='cnns')
    model.summary()
    return model
'''
input: query features and data features
query features: 支持度、下标、范围[将负载特征也用词向量进行编号] 
data features: 对1000条数据词嵌入的结果 1000*40*50
I will use CNN to extract data features in order to decrease paramters
'''

tf.enable_eager_execution()
filter_sizes=[5,10,20,30,50]
embedding_dimension=26
sequence_length = 30000
alpha_len = 7

model = keras.Sequential([
        layers.Input(shape=([sequence_length,alpha_len,embedding_dimension])),
        layers.Reshape((sequence_length,alpha_len,embedding_dimension,1)),
        convolution(),
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(4, activation='sigmoid')
    ])
model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
model.summary()
'''
x_test = np.loadtxt("strtest.csv", dtype=np.str,delimiter=',')
x_test = Dataset().dataset_read(x_test,98)
y_test = np.loadtxt("testout.csv", delimiter=',')
'''
model.load_weights("./2/weight")
test = np.loadtxt("test1.csv", dtype=np.str,delimiter=',')
test = Dataset().dataset_read(test,8)
y = model.predict(test)
print(y)
strtrain = np.loadtxt("train.csv", dtype=np.str,delimiter=',')
input = Dataset().dataset_read(strtrain,2048)
#input = np.fromfile("onehot.dat", dtype=np.int8, sep=",").reshape(1792, sequence_length, alpha_len, embedding_dimension)
out = np.loadtxt("out.csv", delimiter=',')
output = out
X_train,x_test , y_train, y_test = train_test_split(input, output, test_size=0.2)
loss = []
acc = []
tloss = []
tacc = []
for i in range(50):
    history = model.fit(X_train, y_train)
    loss.append(history.history['accuracy'][0])
    acc.append(history.history['loss'][0])
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    tloss.append(test_loss)
    tacc.append(test_accuracy)

model.save_weights("./2/weight")
plt.plot(acc)
plt.plot(loss)
plt.plot(tacc)
plt.plot(tloss)
#plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'train loss','testing','test loss'], loc='lower right')
plt.show()
#model.load_weights("./2/weight")

y_hat = model.predict(x_test)
print(y_hat)
test_loss, test_accuracy = model.evaluate(x_test,y_test)
print(test_accuracy)
print(test_loss)
'''
tloss = []
taccu = []
for i in range(0,100):
    history = model.fit(X_train, y_train)
    #prediction = model.predict(X_test)
    #prediction = np.ceil(prediction)
    #model.save_weights("weight")
    test_loss, test_accuracy = model.evaluate(X_test,y_test)
    tloss.append(test_loss)
    taccu.append(test_accuracy)

plt.plot(np.linspace(0,100,100),tloss)
plt.plot(np.linspace(0,100,100),taccu)
plt.show()
model.save_weights("./2/weight")
#model = tf.keras.models.load_model('all_model.h5')

'''