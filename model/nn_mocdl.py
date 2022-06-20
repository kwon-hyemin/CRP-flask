# 모델은 함수라서 메소드로 처리
import os
import sys
from unittest import result
from sympy import im
from torch import t
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
from icecream import ic

import numpy as np


class Solution:
    def __init__(self) -> None:
        # [털, 날개]
        self.x_data = np.array([[0, 0],[1, 0],[1, 1],[0, 0],[0, 0], [0, 1]])
        #기타, 포유류, 조류
        # 원핫 인코딩
        self.y_data = np.array([
        [1, 0, 0], # 기타
        [0, 1, 0], # 포유류
        [0, 0, 1], # 조류
        [1, 0, 0], # 기타
        [0, 1, 0], # 포유류
        [0, 0, 1] ]) #조류
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        self.W = tf.Variable(tf.random_uniform([2, 3], -1, 1.))
        self.b = tf.Variable(tf.zeros([3]))
        self.L = tf.add(tf.matmul(self.X, self.W), self.b)
        self.model = tf.nn.softmax(self.L)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.model), axis = 1))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train_op = self.optimizer.minimize(self.cost)
    def hook(self):
        def print_menu():
            print('0. Exit')
            print('1. 신경망 모델 구성')
            print('2. 신경망 학습 모델')
            print('3. 결과확인')
            return input('메뉴 선택 \n')

        while 1:
            menu = print_menu()
            if menu == '0':
                self.hook()
            elif menu == '1':
                self.placholder_result()
            elif menu == '2':
                pass
            elif menu == '3':
                pass
            elif menu == '0':
                break
    
    

    def train_nn_model(self):
        
        # **********
        # 신경망 학습 모델
        # **********
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in range(100):
            sess.run(self.train_op, {self.X: self.x_data, self.Y:  self.y_data})
            if (step + 1) % 10 == 10:
                print(step +1, sess.run(self.cost, {self.X: self.x_data, self.Y: self.y_data}))

        return sess

    def placholder_result(self):
        sess = self.train_nn_model()
        # *********
        # 결과확인
        # ********
        prediction = tf.argmax(self.model, 1)
        target = tf.argmax(self.Y, 1)
        print('예측값', sess.run(prediction, {self.X: self.x_data}))
        print('실제값', sess.run(target, {self.Y: self.y_data}))
        # tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옴
        # 예) [[0, 1, 0][1, 0, 0]] -> [1, 0]
        #  [[0.2, 0.7, 0.1][0.9, 0.1, 0.]] -> [1, 0]
        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도: %.2f' % sess.run(accuracy * 100, {self.X: self.x_data, self.Y: self.y_data}))








if __name__ == '__main__':
    ic(basedir)
    tf.disable_v2_behavior()
    ic(tf.__version__)
    s = Solution()
    s.hook()