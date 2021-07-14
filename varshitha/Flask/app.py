# importing the necessary dependencies
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

# initializing the flask app
app = Flask(__name__)
filepath=r"C:\Users\JYOTHSNA\Downloads\CO2-Emission-of-Cars-main\Flask\CO2.pkl"
model = pickle.load(open(filepath,'rb'))

# route to display the home page
@app.route('/')
def home():
    return render_template('home.html')  # rendering the home page


# route which will take you to the prediction page
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')


# route to show the predictions in a web UI
@app.route('/predict',methods=["POST","GET"])
def predict():
    
    # reading inputs from the user
    input_feature=[float(x) for x in request.form.values()]
    features_values=[np.array(input_feature)]
    feature_name=['Make','Vehicle_Class','Engine_Size','Cylinders','Transmission','Fuel_type','Fuel_Consumption_City','Fuel_Consumption_Hwy','Fuel_Consumption_Comb(mpg)']
    x = pd.DataFrame(features_values,columns=feature_name)
    
    # predictions using the loaded model file
    prediction=model.predict(x)
    print("Prediction is:",prediction)
    
    # showing the prediction results in a UI
    return render_template("resultnew.html",prediction=prediction[0])
if __name__ == "__main__":
    
    app.run(debug=True)    # running the app
    