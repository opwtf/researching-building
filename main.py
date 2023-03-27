import pickle
import tensorflow
from tensorflow import keras
from keras.models import model_from_json
import flask
from flask import Flask
from flask import render_template
import sklearn

app = Flask(__name__, template_folder='templates')
@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template("index.html") #получения стартовой старницы
    if flask.request.method == 'POST':
        json_file = open('mod.json', 'r')
        loaded_model_json = json_file.read() #загрузка архитектуры нейронной сети
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('mod.h5') #загрузка весов предобученной нейронной сети
        with open('knn_br_model.pkl', 'rb') as f:#загррузка модели К-ближайших соседей
            loaded_model_knn = pickle.load(f)
        with open('bag_br_model.pkl', 'rb') as f:#загрузка модели ансамбля бэггинг
            loaded_model_bag = pickle.load(f)
        # запись значений из формы в переменные
        exp1 = float(flask.request.form["Соотношение матрица-наполнитель"])
        exp2 = float(flask.request.form["Плотность, кг/м3"])
        exp3 = float(flask.request.form["модуль упругости, ГПа"])
        exp4 = float(flask.request.form["Количество отвердителя, м.%"])
        exp5 = float(flask.request.form["Содержание эпоксидных групп,%_2"])
        exp6 = float(flask.request.form["Температура вспышки, С_2"])
        exp7 = float(flask.request.form["Поверхностная плотность, г/м2"])
        exp8 = float(flask.request.form["Потребление смолы, г/м2"])
        exp9 = float(flask.request.form["Угол нашивки, град"])
        exp10 = float(flask.request.form["Шаг нашивки"])
        exp11 = float(flask.request.form["Плотность нашивки"])
        exp12 = 0
        exp15 = float(flask.request.form["Модуль упругости при растяжении, ГПа"])
        exp16 = float(flask.request.form["Прочность при растяжении, МПа"])
        #выделение дополнительной переменной, для классификации
        if exp3 >= 750:
            exp12 = 0
        else:
            exp12 = 1
        #запись переменных и нормализация для подачи в нейронную сеть
        mat1 = (exp2 - 1731.764) / 476.009
        mat2 = (exp3 - 2.436) / 1909.1
        mat3 = (exp4 - 17.740) / 181.213
        mat4 = (exp5 - 14.254) / 18.746
        mat5 = (exp6 - 100) / 313.273
        mat6 = (exp7 - 0.603) / 1398.939
        mat7 = (exp8 - 64.054) / 18.628
        mat8 = (exp15 - 1036.858) / 2811.58
        mat9 = (exp16 - 33.803) / 380.787
        mat10 = (exp9 - 0) / 90
        mat11 = (exp10 - 0) / 14.44
        mat12 = (exp11 - 0) / 103.98
        mat13 = 0
        if mat2 >= 750:
            mat13 = 0
        else:
            mat13 = 1
        # запись данных в думерные массивы
        X_net = [[mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9, mat10, mat11, mat12, mat13]]
        X = [[exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10,exp12]]
        #предсказание данных
        y_pre_norm = loaded_model.predict(X_net)
        #денормализация
        y_pre = y_pre_norm * 5.202 + 0.389
        y_pred = loaded_model_knn.predict(X)
        y_pred_bag = loaded_model_bag.predict(X)
        if exp1==0:
            return render_template("index.html",result='no', result1='no', result2=y_pre)
        if ((exp15==0)and(exp16==0)):
            return render_template("index.html", result=y_pred, result1=y_pred_bag,result2='no')
        else:
          return render_template("index.html", result=y_pred, result1=y_pred_bag, result2=y_pre)


if __name__ == '__main__':
    app.run()
