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


@app.route('/prediction', methods = ['GET', 'POST'] )
def prediction():

    if model:
        if request.get_json() is not None:
            json_ = request.json
            pred1 = model.predict([json_['image1']])
            pred2 = model.predict([json_['image2']])
            
            if pred1 == pred2:
                return jsonify({'Result':'Both images are same'})
            else:
                return jsonify({'Result':"Both images are different"})
                                

    else:
        print ('No Model available')
        return jsonify({'Error':'No model available please train the model'})


if __name__ == '__main__':
  
    app.run(host='0.0.0.0', port=8080, debug=True)