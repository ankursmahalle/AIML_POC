from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import flask_monitoringdashboard as dashboard
import pandas as pd
import os
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import get

from apps.training.train_model import TrainModel
from apps.prediction.predict_model import PredictModel
from apps.core.config import Config
import matplotlib.pyplot as plt
import seaborn as sns
import orjson

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)

#@app.route('/', methods=['POST','GET'])
#def index_page():
#    return render_template('index.html')

@app.route('/', methods=['GET'])
def index_page():
    return render_template('response.html')

@app.route('/training', methods=['POST'])
@cross_origin()
def training_route_client():
    dp = pd.read_csv("C:\\Users\\MCS\\Downloads\\diabetes.csv")
    print(dp.head())
    sns.countplot(x='Outcome', data=dp)
    plt.show
    try:
        config = Config()
        # get run id
        run_id = config.get_run_id()
        data_path = config.training_data_path
        # trainmodel object initialization
        trainModel = TrainModel(run_id, data_path)
        # training the model
        trainModel.training_model()
        return Response("Training successfull! and its RunID is : " + str(run_id))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

@app.route('/batchprediction', methods=['POST'])
@cross_origin()
def batch_prediction_route_client():
    """
    * method: batch_prediction_route_client
    * description: method to call batch prediction route
    * return: none
    *   None
    """
    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        #prediction object initialization
        predictModel=PredictModel(run_id, data_path)
        #prediction the model
        predictModel.batch_predict_from_model()
        return Response("Prediction successfull! and its RunID is : "+str(run_id))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route('/prediction', methods=['POST'])
@cross_origin()
def single_prediction_route_client():
    """
    * method: prediction_route_client
    * description: method to call prediction route
    * return: none

    """
    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        print('Test')

        if request.method == 'POST':
            Pregnancies = request.form['Pregnancies']
            Glucose = request.form["Glucose"]
            BloodPressure = request.form["BloodPressure"]
            SkinThickness = request.form["SkinThickness"]
            Insulin = request.form["Insulin"]
            BMI = request.form["BMI"]
            DiabetesPedigreeFunction = request.form["DiabetesPedigreeFunction"]
            Age = request.form["Age"]

            data = pd.DataFrame(data=[[Pregnancies, Glucose, BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]],
                              columns=['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
            # using dictionary to convert specific columns
            #convert_dict = {'Pregnancies': int,
             #               'Glucose': int,
             #               'BloodPressure': int,
             #               'SkinThickness': int,
             #               'Insulin': int,
             #               'BMI': float,
             #               'DiabetesPedigreeFunction': float,
             #               'Age': object}
            #data = data.astype(convert_dict)
            data = data.convert_dtypes()


            # object initialization
            predictModel = PredictModel(run_id, data_path)
            # prediction the model
            output = predictModel.single_predict_from_model(data)
            print('output : '+str(output))
            return Response("Predicted Output is : "+str(output))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

@app.route('/predictions', methods=['GET'])
@cross_origin()
def singles_prediction_route_client():
    """
    * method: prediction_route_client
    * description: method to call prediction route
    * return: none

    """
    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        print('Test')

        if request.method == 'GET':
            data = pd.read_csv("C:\\Users\\MCS\\Downloads\\diabetes.csv")
            #print(data.head())
            print(data)
            #sns.countplot(x='Outcome', data=data)
            #plt.show
            #data.cast=data.cast.apply(orjson.loads)

            #print('output : ')
            #print(data.decode('utf-8'))
            #data = pd.DataFrame(data=[["Pregnancies", "Glucose","Insulin","Age"]],
                              #  columns=[Pregnancies','Glucose', 'Insulin','Age'])
            temp=data.loc[:, ["Pregnancies", "Glucose","Insulin","Age"]].to_json(orient='records')
            #creates json dumps
            temp = orjson.dumps(temp)
            return Response(temp)
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    #app.run()
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()