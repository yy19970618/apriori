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

def initConv(shape,dtype=None):
    res = np.zeros(shape)
    res[0], res[-1] = 1, 1
    return tf.constant_initializer(res)
def convolution():

    inn = layers.Input(shape=(sequence_length,1))
    cnns = []
    for i,size in enumerate(filter_sizes):
        conv = layers.Conv1D(filters=8, kernel_size=(size),
                            strides=size, padding='valid', activation='relu',kernel_initializer=initConv([size,8]))(inn)
        #if i%2:
        pool_size =int(conv.shape[1]/100)
        pool = layers.MaxPool1D(pool_size=(pool_size), padding='valid')(conv)
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
filter_sizes=[5,50]
embedding_dimension=50
sequence_length = 30000
'''
#单分类，效果还不错，正确率接近1
model = keras.Sequential([
        layers.Input(shape=([sequence_length])),
        layers.Reshape((sequence_length,1)),
        convolution(),
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
'''
model = keras.Sequential([
        layers.Input(shape=([sequence_length])),
        layers.Reshape((sequence_length,1)),
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
model.load_weights("./1/weight")
x_test = np.loadtxt("test.csv", delimiter=',')
y_test = np.loadtxt("testout.csv", delimiter=',')
y_hat = model.predict(x_test)
print(y_hat)
test_loss, test_accuracy = model.evaluate(x_test,y_test)
print(test_accuracy)
print(test_loss)
'''
input = np.loadtxt("numtrain.csv",delimiter=',')
out = np.loadtxt("out1.csv", delimiter=',')
input = sp.scale(input)
output = out
X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2)

history = model.fit(X_train, y_train,epochs=500,validation_split=0.1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])

plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'loss','valiation'], loc='upper left')

plt.show()

tloss = []
taccu = []
for i in range(0,200):
    history = model.fit(X_train, y_train)
    #prediction = model.predict(X_test)
    #prediction = np.ceil(prediction)
    #model.save_weights("weight")
    test_loss, test_accuracy = model.evaluate(X_test,y_test)
    tloss.append(test_loss)
    taccu.append(test_accuracy)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()

plt.plot(np.linspace(0,200,200),tloss)
plt.plot(np.linspace(0,200,200),taccu)
plt.show()
model.save_weights("./1/weight")
#model = tf.keras.models.load_model('all_model.h5')



model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])

model.save('all_model.h5')
model = tf.keras.models.load_model('all_model.h5')

print(tf.__version__)

model = keras.models.Sequential([
    keras.layers.Dense([100], input_shape=[1000,40,50])
])
# tensorboard earlystopping
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "index_model.h5")

callbacks =[
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5,
                                  min_delta=1e-3)
]
'''