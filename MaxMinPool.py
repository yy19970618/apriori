
import tensorflow as tf
import numpy as np

import tensorflow.python.keras.layers as layers


# 定义网络层就是：设置网络权重和输出到输入的计算过程
'''
class MaxMin(layers.Layer):
    def __init__(self,poolsize):
        super(MaxMin, self).__init__()
        self.poolsize = poolsize

    def call(self, inputs):
        #input is a tensor
        dim = inputs.shape.as_list()[0]
        res = tf.constant([0],dtype=float)
        bar = int(np.floor(dim/self.poolsize))
        for i in range(0, bar):
            a = 0 + i*self.poolsize
            slice = tf.slice(inputs,[a],[self.poolsize])
            sortedslice = tf.sort(slice)
            s = tf.stack([sortedslice[0],sortedslice[-1]],axis=0)
            res = tf.concat([res, s], axis=0)
        return tf.slice(res,[1],[bar*2])
'''
class MaxMin(layers.Layer):
    def __init__(self,poolsize):
        super(MaxMin, self).__init__()
        self.poolsize = poolsize

    def call(self, inputs):
        #input is a tensor

        feature_map = inputs
        channel = feature_map.shape[0]
        height = feature_map.shape[1]
        outdim = height/self.poolsize
        pool_out = tf.zeros([channel, 2*int(outdim)])
        for map_num in range(channel):
            out_height = 0
            for r in np.arange(0, height, self.poolsize):
                a = tf.argmax(feature_map[map_num, r:r + self.poolsize, ])
                pool_out[map_num, out_height] = tf.argmax(feature_map[map_num, r:r + self.poolsize,])
                pool_out[map_num, out_height+1] = tf.minimum(feature_map[map_num, r:r + self.poolsize,])
                out_height = out_height + 2

        return pool_out

x = tf.constant([[999,342,80,79,105,30,43,26, 78,40],
                 [999, 342, 80, 79, 105, 30, 43, 26, 78, 40],
                 [999, 342, 80, 79, 105, 30, 43, 26, 78, 40]],dtype=float)

my_layer = MaxMin(3)
out = my_layer(x)
print(out)
