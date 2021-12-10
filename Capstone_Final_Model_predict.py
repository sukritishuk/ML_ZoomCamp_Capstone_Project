# Loading the Model - 
import pickle

from flask import Flask
from flask import request
from flask import jsonify
from sklearn.ensemble import RandomForestClassifier


# creating a variable with our model file:
input_file = 'model.bin'

# loading our model file: 
with open(input_file, 'rb') as f_in:    # file input; rb - used to read the file
    dv, model = pickle.load(f_in)     # load() function reads from the file

# creating a flask app -
app = Flask(__name__)

# adding a decorator to our function -
@app.route('/collision_injury_predict/', methods=['POST'])


def collision_injury_predict():
    # specifying request to be in JSON format converted to Python dictionary:
    sample_collided_person = request.get_json()

    # transforming the sample_collided_person's feature details into a dictionary using DictVectorizer:
    X = dv.transform([sample_collided_person])

    # make prediction on sample_collided_person using our model: 
    y_pred = model.predict_proba(X)[:,1]

    # specifying the collision_injury decision for our model by specifying the threshold >= 0.55:
    collision_injury = float(y_pred) >= 0.55 

    # specify what response we want the web service to return to us:
    result = {'collision_injury_prediction': float(y_pred), 'collision_injury': bool(collision_injury)}

    return jsonify(result)


if __name__== "__main__":
    # running the commands in debug mode, specifying the host and port -
    app.run(debug=True)