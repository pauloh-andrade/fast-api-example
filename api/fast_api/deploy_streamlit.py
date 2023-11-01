import os
import pickle

import numpy as np
import pandas as pd

import streamlit as st


st.set_option('deprecation.showPyplotGlobalUse', False)
model = pickle.load(open(os.path.join(os.getcwd(), 'finalized_model.pkl'), 'rb'))


def logistic_regression(evaluation: int) -> float:

    input = np.array(evaluation).reshape(1, -1)
    prediction = model.predict(input)

    return prediction


def main():

    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Classification Model </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    x1 = st.number_input('x1')
    x2 = st.number_input('x2')
    x3 = st.number_input('x3')
    x4 = st.number_input('x4')

    X = [x1, x2, x3, x4]
    st.write('The current number is ',  X)
    prediction = logistic_regression(X)

    if st.button("Rodar"):
        st.success(prediction[0])


if __name__ == '__main__':
    main()