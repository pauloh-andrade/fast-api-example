from fastapi import FastAPI
import os
import pickle

import numpy as np

import streamlit as st


model_api = FastAPI()


st.set_option('deprecation.showPyplotGlobalUse', False)
model = pickle.load(open(os.path.join(os.getcwd(), 'finalized_model.pkl'), 'rb'))


def logistic_regression(evaluation: list) -> float:

    input = np.array(evaluation).reshape(1, -1)
    prediction = model.predict(input)

    return prediction

@model_api.get('/index')
async def home():
    return "Hello WOrld"

@model_api.get("/model")
async def home_home(text: str) -> dict:
    return { "home": text }

@model_api.post("/post")
async def home_home(x1: int, x2: int, x3: int, x4: int) -> str:
    predict = logistic_regression([x1, x2, x3, x4])

    return predict[0]