import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
import os
import sys
from unittest import result
from sympy import im
from torch import t
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
from icecream import ic
from matplotlib import rc, font_manager
rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/malgunsl.ttf').get_name())
import PyQt5

class Solution():
    def __init__(self) -> None:
        pass
    def train_nn_model(self):
        num_points = 1000
        vectors_set = []

        for i in range(num_points):
            x1 = np.random.normal(0.0, 0.55)
            y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
            vectors_set.append([x1, y1])  # 차원누락

        x_data = [v[0] for v in vectors_set]
        y_data = [v[1] for v in vectors_set]

        W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1])) # zeros 는 배열내부를 0으로 초기화하라
        y = W * x_data + b
        loss = tf.reduce_mean(tf.square(y - y_data)) # 경사하강법

        optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5 는 학습속도
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer() # 변수초기화. 텐서플로는 반드시 변수초기화 필요
        sess = tf.Session()
        sess.run(init)

        for i in range(8):
            sess.run(train)
            print(sess.run(W), sess.run(b))
            plt.plot(x_data, y_data, 'ro')
            plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if __name__=='__main__':
    ic(basedir)
    tf.disable_v2_behavior()
    s = Solution()
    s.train_nn_model()