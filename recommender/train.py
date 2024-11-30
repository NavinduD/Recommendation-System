import tensorflow as tf
from recommender.preprocess import preprocess_data

def train_model():
    # Preprocess data
    train_data, test_data = preprocess_data("recommender/data/sales_report.csv")
    num_users = train_data['customer_id'].nunique()
    num_items = train_data['style_id'].nunique()
    embedding_size = 50

    # Build the model
    class RecommenderModel(tf.keras.Model):
        def __init__(self, num_users, num_items, embedding_size):
            super().__init__()
            self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
            self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
            self.dot = tf.keras.layers.Dot(axes=1)

        def call(self, inputs):
            user_vec = self.user_embedding(inputs[:, 0])
            item_vec = self.item_embedding(inputs[:, 1])
            return self.dot([user_vec, item_vec])

    model = RecommenderModel(num_users, num_items, embedding_size)
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    train_inputs = train_data[['customer_id', 'style_id']].values
    train_labels = train_data['rate_scaled'].values
    model.fit(train_inputs, train_labels, batch_size=32, epochs=10)

    # Save the model
    model.save("recommender/model/recommender_model.h5", save_format="h5")

if __name__ == "__main__":
    train_model()
