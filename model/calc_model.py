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
from matplotlib import rc, font_manager
rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/malgunsl.ttf').get_name())
class CalculatorModel:
    def __init__(self) -> None:
        self.model = os.path.join(basedir, 'model')
        self.data = os.path.join(self.model, 'data')

    def calc(self, num1, num2, opcode):
        print(f'훅에 전달된 num1 : {num1}, num2 : {num2}, opcode : {opcode}')
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.import_meta_graph(self.model + '/calculator_'+opcode+'/model-1000.meta')
            graph = tf.get_default_graph()
            w1 = graph.get_tensor_by_name('w1:0')
            w2 = graph.get_tensor_by_name('w2:0')
            feed_dict = {w1: float(num1), w2: float(num2)}
            op_to_restore = graph.get_tensor_by_name('op_'+opcode+':0')
            result = sess.run(op_to_restore, feed_dict)
            print(f'최종결과: {result}')
        return result

    def create_add_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 5.0, 'w2': 2.0}
        r = tf.add(w1, w2, name='op_add')
        sess = tf.Session() # <- 1버전  model = tf.keras.models.Sequential([ <- 2버전
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] : {feed_dict['w1']}")
        print(f"feed_dict['w2'] : {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 덧셈결과: {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_add', 'model'), global_step=1000)




    def create_sub_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 6.0, 'w2': 2.0}
        r = tf.subtract(w1, w2, name='op_sub')
        sess = tf.Session() # <- 1버전  model = tf.keras.models.Sequential([ <- 2버전
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] : {feed_dict['w1']}")
        print(f"feed_dict['w2'] : {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 뺄셈 결과: {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_sub', 'model'), global_step=1000)

    def create_mul_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 7.0, 'w2': 2.0}
        r = tf.multiply(w1, w2, name='op_mul')
        sess = tf.Session() # <- 1버전  model = tf.keras.models.Sequential([ <- 2버전
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] : {feed_dict['w1']}")
        print(f"feed_dict['w2'] : {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 곱셈 결과: {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_mul', 'model'), global_step=1000)


    def create_div_model(self):
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}
        r = tf.divide(w1, w2, name='op_div')
        sess = tf.Session() # <- 1버전  model = tf.keras.models.Sequential([ <- 2버전
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(f"feed_dict['w1'] : {feed_dict['w1']}")
        print(f"feed_dict['w2'] : {feed_dict['w2']}")
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'TF 나눗셈 결과: {result}')
        saver.save(sess, os.path.join(self.model, 'calculator_div', 'model'), global_step=1000)

if __name__=='__main__':
    ic(basedir)
    tf.disable_v2_behavior()
    ic(tf.__version__)
    hello = tf.constant("Hello")
    session = tf.Session()
    ic(session.run(hello))
    c = CalculatorModel()
    # c.create_add_model()
    # c.create_sub_model()
    # c.create_mul_model()
    c.create_div_model()