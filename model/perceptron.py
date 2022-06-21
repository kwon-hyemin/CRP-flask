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
class Solution():
    def __init__(self) -> None:
        tf.set_random_seed(777)
        X = [1, 2, 3]
        Y = [1, 2, 3]
        self.W = tf.placeholder(tf.float32)
        hypothesis = X * self.W

        self.cost = tf.reduce_mean(tf.square(hypothesis - Y))
        self.sess = tf.Session()

    def train_nn_model(self):
        W_history = []
        cost_history = []
        for i in range(-30, 50):
            curr_W = i * 0.1
            curr_cost = self.sess.run(self.cost, {self.W: curr_W})
            W_history.append(curr_W)
            cost_history.append(curr_cost)
        # 차트로 확인
        plt.plot(W_history, cost_history)
        plt.show()

if __name__=='__main__':
    ic(basedir)
    tf.disable_v2_behavior()
    s = Solution()
    s.train_nn_model()