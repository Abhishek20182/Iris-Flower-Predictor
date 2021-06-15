#Python Libraries 
from flask import Flask, request, jsonify
import flask
import pickle
import joblib
import numpy as np

#ref: https://www.wintellect.com/creating-machine-learning-web-api-flask/

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')

# Load the Random Forest CLassifier model
filename = 'model.pkl'
clf = joblib.load(open(filename, 'rb'))

@app.route("/predict", methods=['POST'])
def predict():
	if request.method == "POST":
		sepal_length = float(request.form['sepal_length'])
		sepal_width = float(request.form['sepal_width'])
		petal_length = float(request.form['petal_length'])
		petal_width = float(request.form['petal_width'])

		data = np.array([[sepal_length, sepal_width, petal_length, petal_width]]).reshape(1, -1)
		my_prediction = clf.predict(data)

		return flask.render_template('result.html', prediction=my_prediction)
	else:
		return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True)


