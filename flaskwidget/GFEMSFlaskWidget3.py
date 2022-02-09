#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
$ export FLASK_APP=GFEMSFlaskWidget3.py
$ flask run
 * Running on http://127.0.0.1:5000/
'''
#source venv/bin/activate
from flask import Flask, render_template, request, redirect,url_for
from flask_wtf import FlaskForm
from flask_classy import FlaskView
import numpy as np
import pickle
from wtforms import Form, FloatField, validators, StringField, SubmitField, IntegerField, RadioField#, NumberRange
import wtforms
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'sd89w7w3r23v145242dsf9089erfm077'

class ShannonForm(FlaskForm):
    lat = FloatField('Latitude', validators=[validators.DataRequired(message="Requires a float"), validators.NumberRange(min=-90, max=90, message="Requires a number between -90 and 90. Southern latitudes should have a negative value. Include as many decimals as you have.")])
    lng = FloatField('Longitude', validators=[validators.DataRequired(message="Requires a float"), validators.NumberRange(min=-180, max=180, message="Requires a number between -180 and 180. Negative longitudes are those west of the prime meridian. Include as many decimals as you have.")])
    authorised_capital = IntegerField('Authorized Capital', validators=[validators.DataRequired(message="Requires an integer.")])
    paid_up_capital = IntegerField('Paid Up Capital', validators=[validators.DataRequired(message="Requires an integer.")])
    trade_partner_count = IntegerField('Number of International Trade Partners', validators=[validators.InputRequired(message="Requires an integer. If unknown, put 0.")])

class ModelPicker(FlaskForm):
    choice = wtforms.fields.SelectField('Model Choice ', choices=[("knn", "KNearestNeighbors"),("rf", "Random Forest")])
    #dropdown_list = ['KNN', 'Random Forest']



@app.route('/home', methods=["GET","POST"])
def view_method():
    '''
    myview = ModelPicker()
    if myview.validate_on_submit():
        inputdata = myview.dropdown_list(inputdata)
        '''

    choiceForm = ModelPicker()
    if choiceForm.validate_on_submit():
        choice = choiceForm.choice.data
        if choice == "knn":
            return redirect(url_for('formKNN'))
        else:
            return redirect(url_for('formRF'))

    return render_template("home.html", form=choiceForm)
    # dropdown_list = ['KNN', 'Random Forest']
    # if flask.request.method == "GET":
    #     return render_template('home.html', dropdown_list=dropdown_list)
    # elif flask.request.method == "POST":
    #     # get dropdown value

    #     # if knn redirect to knn, if random redict there

@app.route('/formKNN', methods=["GET", "POST"])
def formKNN():
    aform = ShannonForm()
    if aform.validate_on_submit():
        classifier, scaler = pickle.load(open('knnendstate.p', 'rb'))
        #print([aform.namsxkd.data, aform.lhdn.data, aform.tsld.data, aform.ld11.data, aform.nganh_kd.data])
        inputdata = np.array([[aform.lat.data, aform.lng.data, aform.authorised_capital.data, aform.paid_up_capital.data, aform.trade_partner_count.data]])
        inputdata = scaler.transform(inputdata)
        prediction = classifier.predict(inputdata)
        if prediction == 0:
            prediction = "low risk"
        if prediction == 1:
            prediction = "high risk"

        message = f'The form has been submitted. You input geo coordinates {aform.lat.data}, {aform.lng.data}, authorized capital {aform.authorised_capital.data}, paid up capital {aform.paid_up_capital.data}, and {aform.trade_partner_count.data} international trade partners. The model predicted that this business is {prediction}.'
        
        
        return render_template('form.html', form=aform, message=message, use_form="formKNN")

    return render_template('form.html', form=aform, use_form="formKNN")

@app.route('/formRF', methods=["GET", "POST"])
def formRF():

    aform = ShannonForm()
    if aform.validate_on_submit():
        classifier, scaler = pickle.load(open('rfendstate.p', 'rb'))
        #print([aform.namsxkd.data, aform.lhdn.data, aform.tsld.data, aform.ld11.data, aform.nganh_kd.data])
        inputdata = np.array([[aform.lat.data, aform.lng.data, aform.authorised_capital.data, aform.paid_up_capital.data, aform.trade_partner_count.data]])
        inputdata = scaler.transform(inputdata)
        prediction = classifier.predict(inputdata)
        if prediction == 0:
            prediction = "low risk"
        if prediction == 1:
            prediction = "high risk"

        message = f'The form has been submitted. You input geo coordinates {aform.lat.data}, {aform.lng.data}, authorized capital {aform.authorised_capital.data}, paid up capital {aform.paid_up_capital.data}, and {aform.trade_partner_count.data} international trade partners. The model predicted that this business is {prediction}.'
        
        return render_template('form.html', form=aform, message=message, use_form="formRF")
        

    return render_template('form.html', form=aform, message=None, use_form="formRF")    

@app.route('/results')
def results():
    bform = ShannonResults()
    if bform.validate_on_submit():
        return bform.outcome
    return render_template('results.html', form = bform)

if __name__ == '__main__':
    app.run(debug=True)

