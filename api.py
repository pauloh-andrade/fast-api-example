from fastapi import FastAPI

model_api = FastAPI()

@model_api.get('/index')
async def home():
    return "Hello WOrld"

@model_api.get("/model")
async def home_home(text: str) -> dict:
    return { "home": text }