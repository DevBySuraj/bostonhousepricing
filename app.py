import pandas as pd
import numpy as np
from flask import Flask,request,app,jsonify,url_for,render_template
import pickle

# make a flask app
app = Flask(__name__) #starting point of the app

#load the model - wb when writing to a file, rb when reading/loading the file
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl','rb'))


#for sending the request to the app through API postman and getting the output
@app.route('/')
def home():
    return render_template('home.html')


#we will give the input from here will get captured by this func and will feed it to the model, postman will be used
#when we call this function the input will be captured in the data variable in json format in the 'data' key
@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    #getting the data values form json(key value pair in a list of single row of multiple features)
    data = np.array(list(data.values())).reshape(1,-1)
    print(data) 

    #standardizing the input data
    new_data = scaler.transform(data)

    #predicing the output ->feeding the transformed input
    output = regmodel.predict(new_data)

    #return 2d array [[30.4]], need for first value only
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()] #forgot the () gave an error
    final_input  = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The Predicted House price is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)


