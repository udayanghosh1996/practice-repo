from flask import Flask, jsonify, request
from joblib import load
  
# creating a Flask app
model_path = 'svm_gamma=0.0004_C=3.0.joblib'
model = load(model_path)


app = Flask(__name__)
  

@app.route('/', methods = ['GET'])
def hello_world():
    if(request.method == 'GET'):
  
        data = "hello world"
        return jsonify({'data': data})

@app.route("/sum", methods = ['POST'])
def sum():
    if(request.method == 'POST'):
        x = request.json['x']
        y = request.json['y']
        z = x + y
        return jsonify({'sum' : z})



@app.route("/predict", methods = ['POST'])
def predict():
    img = request.json['image']
    predict = model.predict([img])
    return jsonify({'predicted': predict})


if __name__ == '__main__':
  
    app.run(debug = True)