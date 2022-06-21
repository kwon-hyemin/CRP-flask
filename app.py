import os
import sys
from model.cabbage import Solution
from model.calc_model import CalculatorModel
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask,render_template,request
from icecream import ic
app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return render_template('index.html')

# @app.route("/calc", methods=["post"])
# def calc():
#     num1 = request.form['num1']
#     num2 = request.form['num2']
#     opcode = request.form['opcode']
#     calc = CalculatorModel()
#     result = calc.calc(num1, num2, opcode)
#     render_params = {}
#     render_params['result'] = result
#     return render_template('index.html', **render_params)

@app.route("/")
def hello_world():
    return render_template('cabbage.html')
    
@app.route("/cabbage", methods=["post"])
def cabbage():
    avgTemp = request.form['avg_temp']
    minTemp = request.form['min_temp']
    maxTemp = request.form['max_temp']
    rainFall = request.form['rain_fall']
    ic(f'{avgTemp},{minTemp},{maxTemp},{rainFall}')
    cabbage = Solution()
    result = cabbage.load_model(avgTemp,minTemp,maxTemp,rainFall)
    render_params = {}
    render_params['result'] = result
    return render_template('cabbage.html', **render_params)



if __name__=='__main__':
    print(f'Started Server')
    app.run()