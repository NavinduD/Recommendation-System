import pandas as pd
import numpy as np

def load_encoders():
    customer_encoder = pd.read_pickle("recommender/model/customer_encoder.pkl")
    style_encoder = pd.read_pickle("recommender/model/style_encoder.pkl")
    return customer_encoder, style_encoder

def prepare_inputs(encoded_customer_id, item_ids):
    return np.array([[encoded_customer_id, item_id] for item_id in item_ids])
