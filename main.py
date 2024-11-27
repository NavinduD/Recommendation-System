from fastapi import FastAPI
import tensorflow as tf # type: ignore
print(tf.__version__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}