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
class GradientDescent():
    def execute(self):
        X = [1., 2., 3,]
        Y = [1., 2., 3,]
        m = n_samples = len(X)
        W = tf.placeholder(tf.float32)
        hypothesis = tf.multiply(X, W)
        cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m
        W_val = []
        cost_val = []
        with tf.Session() as sess:
            #saver2 = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            #saver2.save(sess, "saved/gradient_descent")
            for i in range(-30, 50):
                W_val.append(i * 0.1)
                cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))
            plt.plot(W_val, cost_val, 'ro')
            plt.ylabel('COST')
            plt.xlabel('W')
            plt.savefig("static/img/result.svg")
            print('경사하강법 실행 중 ..')
            return "경사하강법 (Gradient Descent)"