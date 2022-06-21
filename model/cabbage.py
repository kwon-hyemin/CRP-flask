import os
from pickletools import optimize
import sys
from turtle import shape
from unittest import result
from sympy import im
from torch import t
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
from icecream import ic
from matplotlib import rc, font_manager
import numpy as np
import pandas as pd
rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/malgunsl.ttf').get_name())

class Solution():
    def __init__(self) -> None:
        self.basedir = os.path.join(basedir, 'model')
        self.df = None
        self.x_data = None
        self.y_data = None

    def preprocessing(self):
        path='./data/price_data.csv'
        self.df = pd.read_csv(path, encoding='UTF-8', thousands=',')
        ic(self.df)
        # year,avgTemp,minTemp,maxTemp,rainFall,avgPrice
        xy = np.array(self.df, dtype=np.float32)
        ic(type(xy)) # <class 'numpy.ndarray'>
        ic(xy.ndim) # xy.ndim: 2  # 차원
        ic(xy.shape) # xy.shape: (2922, 6) # 행렬의 갯수
        self.x_data = xy[:, 1:5] # 슬라이싱 # 해당 날짜의 날씨에 해당하는 기후요소 4개를 변인으로 받는다.
        self.y_data = xy[:, 5] # 인덱싱  # 해당 날짜의 가격을 입력한다.
        ic(self.x_data)
        ic(self.y_data)
    
    def create_nn_model(self):
        # 모델생성
        # 텐서모델 초기화(모델템플릿생성)
        model = tf.global_variables_initializer() 
        # 확률변수 데이터
        # avgTemp,minTemp,maxTemp,rainFall,avgPrice
        self.preprocessing()
        # 선형식(가설)제작 y = Wx+b  tf.metmul 로 한다
        X = tf.placeholder(tf.float32, shape=[None, 4]) # 4는 행렬의 갯수
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4,1]), name="weight") # 표준화 standard  값 indexing
        b = tf.Variable(tf.random_normal([1]), name="bias")   # 정규화 normal  범위 sliceing
        # 가설은 선형식을 말한다 
        hypothesis =tf.matmul(W, X) + b #Wx + b        
        
         # 손실함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        # 최적화알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        # 세션생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # 트레이닝
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                        feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0:
                print('# %d 손실비용: %d'%(step, cost_))
                print('- 배추가격: %d '%(hypo_[0]))
        # 모델저장
        saver = tf.train.Saver() # 저장도 하고 불러도오는                   
        saver.save(sess, os.path.join(self.model, 'cabbage', 'cabbage.ckpt'), global_step=1000)
        print('저장완료')

    # 모델로드
    def load_model(self, avgTemp,minTemp,maxTemp,rainFall,avgPrice):
        # 선형식(가설)제작 y = Wx+b  tf.metmul 로 한다
        X = tf.placeholder(tf.float32, shape=[None, 4]) # 4는 행렬의 갯수
        W = tf.Variable(tf.random_normal([4,1]), name="weight") # 표준화 standard  값 indexing
        b = tf.Variable(tf.random_normal([1]), name="bias")   # 정규화 normal  범위 sliceing
        saver = tf.train.Saver()   #객체를 만들어줌 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt-1000'))
            data = [[avgTemp,minTemp,maxTemp,rainFall],]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print(dict)
        return int(dict[0])

if __name__=='__main__':
    tf.disable_v2_behavior()
    s = Solution()
    s.create_nn_model()
    
    