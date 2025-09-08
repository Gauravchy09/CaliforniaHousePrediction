from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import send_file, send_from_directory
import os

ridge_model = pickle.load(open('models/model.pkl',"rb"))
Standard_scaler = pickle.load(open('models/standard_scaler.pkl', "rb"))

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        MedInc = float(request.form.get('MedInc'))
        HouseAge = float(request.form.get('HouseAge'))
        AveRooms = float(request.form.get('AveRooms'))
        AveBedrms = float(request.form.get('AveBedrms'))
        Population = float(request.form.get('Population'))
        AveOccup = float(request.form.get('AveOccup'))
        Latitude = (request.form.get('Latitude'))
        Longitude = (request.form.get('Longitude'))

        if Latitude == None or Latitude == '':
            Latitude = 34.26
        else:
            Latitude = float(Latitude)
        if Longitude == None or Longitude == '':
            Longitude = -118.49
        else:
            Longitude = float(Longitude)

        new_data_scaled = Standard_scaler.transform([[Latitude, Longitude, MedInc, AveRooms, AveBedrms, Population, AveOccup, HouseAge]])

        value = ridge_model.predict(new_data_scaled)
        value = abs(value[0])

        return render_template('home.html', results = value)
    else:
        return render_template('home.html')

@app.route("/download")
def download():
    file_path = os.path.join(os.getcwd(), "dataset", "housing.csv")
    return send_file(file_path,
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="CaliforniaHousingDataset.csv")
if __name__ == "__main__":
    app.run(host="0.0.0.0")