from flask import Flask, render_template, request
import pickle 
app = Flask(__name__)
import sklearn as sk
import numpy as np
import pandas as pd

# model = pickle.load(open('svc_model.pkl','rb')) 
aircraft_type = pickle.load(open('aircraft.pkl','rb')) 
bay_number = pickle.load(open('bayno.pkl','rb')) 
model = pickle.load(open('svc_model.pkl','rb'))
dfcols = pickle.load(open('dfcol.pkl','rb')) 



@app.route("/")
def home():
    return render_template('index.html',bay_number = bay_number,aircraft_type=aircraft_type )


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
    
        time = int(request.form["time"])
        count = int(request.form["count"])
        
        
        aircraft_type = 'aircraft_type_' + str(request.form["aircraft_type"])
        bay_number = 'bay_number_' + str(request.form["bay_number"])

        p = pd.DataFrame(columns =dfcols)
        p = p.append(pd.Series(0, index=dfcols), ignore_index=True)
        p.at[0,'time']=time
        p.at[0,'count']=count
        p.at[0,bay_number] = 1
        p.at[0,aircraft_type] = 1
        #get prediction
        
        pred = model.predict(p)
        # pred = p[0]
        return render_template("index.html", prediction_text='pred: {}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)