import pandas
import numpy as np
from flask import Flask,render_template,request,jsonify
import pickle
import time

model = pickle.load(open('./model/rdf_model.pkl','rb'))

app = Flask(__name__)

@app.get('/')
def home():
    return render_template("index.html")

@app.post('/predict')
def predict():
    dependents = int(request.json['dependents'])
    education = int(request.json['graduate'])
    selfEmployed = int(request.json['selfEmployed'])
    income = int(request.json['income'])
    loan = int(request.json['loan'])
    term = int(request.json['term'])
    cibil = int(request.json['cibil'])
    rassets = int(request.json['rassets'])
    cassets = int(request.json['cassets'])
    lassets = int(request.json['lassets'])
    bassets = int(request.json['bassets'])
    input_data = np.array([dependents,education,selfEmployed,income,loan,term,cibil,rassets,cassets,lassets,bassets])
    names = ['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']
    data = pandas.DataFrame([input_data],columns=names)
    prediction = model.predict(data)
    print(f"Prediction - {prediction}")
    time.sleep(2.5)
    return jsonify({
        'status': 200,
        'result': int(prediction)
    })

if __name__=='__main__': 
    app.run(debug=True)