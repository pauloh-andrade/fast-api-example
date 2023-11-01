from flask import Flask, request, jsonify

import os
import pickle
import streamlit as st
import numpy as np

app = Flask(__name__)


st.set_option('deprecation.showPyplotGlobalUse', False)
model = pickle.load(open(os.path.join(os.getcwd(), 'finalized_model.pkl'), 'rb'))

@app.route('/prever', methods=['GET'])
def prever():
    x1 = float(request.args.get('x1'))
    x2 = float(request.args.get('x2'))
    x3 = float(request.args.get('x3'))
    x4 = float(request.args.get('x4'))

    entry = np.array([[x1, x2, x3, x4]])

    result = model.predict(entry)

    return jsonify({'predict': result.tolist()})

app.run(debug=True)