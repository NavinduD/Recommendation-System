import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath)

    # Encode customer and style columns
    customer_encoder = LabelEncoder()
    style_encoder = LabelEncoder()
    data['customer_id'] = customer_encoder.fit_transform(data['CUSTOMER'])
    data['style_id'] = style_encoder.fit_transform(data['Style'])

    # Scale numeric columns
    scaler = MinMaxScaler()
    data['rate_scaled'] = scaler.fit_transform(data[['RATE']])

    # Save encoders for reuse
    os.makedirs("recommender/model", exist_ok=True)
    pd.to_pickle(customer_encoder, "recommender/model/customer_encoder.pkl")
    pd.to_pickle(style_encoder, "recommender/model/style_encoder.pkl")
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data
