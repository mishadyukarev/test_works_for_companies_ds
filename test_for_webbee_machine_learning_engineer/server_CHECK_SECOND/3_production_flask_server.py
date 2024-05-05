import pickle

import pandas as pd
from flask import Flask
from flask import request

from io import StringIO


app = Flask(__name__)
with open('best_regressor_pipeline.pkl', 'rb') as pickle_file:
    model_pl = pickle.load(pickle_file)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file = request.files['file']
    file_content = file.read()

    file_str = file_content.decode('utf-8')

    x_for_predict = pd.read_json(StringIO(file_str))
    y_predicted = model_pl.predict(x_for_predict)

    return y_predicted.tolist()


if __name__ == '__main__':
    app.run()

#%%

#%%
