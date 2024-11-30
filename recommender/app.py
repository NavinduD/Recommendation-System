from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from recommender.utils import load_encoders, prepare_inputs

app = FastAPI()

model = tf.keras.models.load_model("recommender/model/recommender_model.h5")
customer_encoder, style_encoder = load_encoders()

@app.get("/recommend/{customer_id}")
def recommend(customer_id: int, num_recommendations: int = 5):
    encoded_customer_id = customer_encoder.transform([customer_id])[0]
    item_ids = np.arange(style_encoder.classes_.shape[0])
    inputs = prepare_inputs(encoded_customer_id, item_ids)
    predictions = model.predict(inputs)
    top_items = item_ids[np.argsort(predictions.flatten())[::-1][:num_recommendations]]
    recommendations = style_encoder.inverse_transform(top_items)
    return {"recommendations": recommendations}
