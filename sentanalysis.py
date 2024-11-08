import tensorflow as tf
from keras.layers import Dense, Input, Dropout, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")


class SentimentAnalyzer:
    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None

    def get_sample_data(self):
        """Create a larger, categorized dataset"""
        # Product Reviews
        positive_product = [
            "This product exceeds all my expectations!",
            "Incredible quality for the price, highly recommended!",
            "Best purchase I've made this year!",
            "Works perfectly, exactly what I needed",
            "Outstanding build quality and excellent performance",
            "Very user-friendly and efficient product",
            "Great value for money, would buy again",
            "This has made my life so much easier",
            "Perfect solution to my needs",
            "Impressive features and reliability",
            "Really happy with this purchase",
            "Superior quality compared to competitors",
            "Excellent product, worth every penny",
            "Fantastic design and functionality",
            "Just what I was looking for",
        ] * 10  # Multiply for more data

        negative_product = [
            "Poor quality, broke after first use",
            "Complete waste of money",
            "Would not recommend to anyone",
            "Terrible design and functionality",
            "Not worth the price at all",
            "Disappointing performance overall",
            "Save your money, buy something else",
            "Failed to meet basic expectations",
            "Poorly made and unreliable",
            "Worst purchase I've ever made",
            "Don't waste your time with this",
            "Horrible customer experience",
            "Product arrived damaged and unusable",
            "Many defects and issues",
            "Regret buying this product",
        ] * 10

        # Service Reviews
        positive_service = [
            "Exceptional customer service!",
            "Very professional and helpful staff",
            "Quick response to all my questions",
            "Wonderful experience with the team",
            "Above and beyond my expectations",
            "Fantastic support throughout",
            "Very satisfied with the service",
            "Prompt and efficient assistance",
            "Great communication and follow-up",
            "Professional and courteous staff",
            "Amazing attention to detail",
            "Service was quick and efficient",
            "Very knowledgeable team",
            "Excellent problem resolution",
            "Would definitely use again",
        ] * 10

        negative_service = [
            "Terrible customer service experience",
            "Never received a response to my inquiry",
            "Very unprofessional staff",
            "Worst service I've ever received",
            "Complete lack of communication",
            "Rude and unhelpful representatives",
            "Extremely slow response time",
            "Poor problem resolution",
            "Incompetent service team",
            "Awful experience overall",
            "No follow-up whatsoever",
            "Frustrating and time-wasting",
            "Completely ignored my concerns",
            "Unhelpful and dismissive",
            "Would never use again",
        ] * 10

        # Restaurant Reviews
        positive_restaurant = [
            "Delicious food and great atmosphere!",
            "Best dining experience ever",
            "Exceptional service and amazing food",
            "Perfect place for special occasions",
            "Outstanding menu selection",
            "Fantastic flavors and presentation",
            "Great value for the quality",
            "Wonderful ambiance and staff",
            "Fresh ingredients and creative dishes",
            "Impressive wine selection",
            "Excellent portion sizes",
            "Beautiful restaurant interior",
            "Perfect temperature and seasoning",
            "Friendly and attentive waitstaff",
            "Will definitely return soon",
        ] * 10

        negative_restaurant = [
            "Terrible food quality",
            "Poor service and cold food",
            "Overpriced for what you get",
            "Dirty restaurant and bad hygiene",
            "Rude staff and long wait times",
            "Disappointing menu options",
            "Food was undercooked",
            "Worst restaurant experience",
            "Would not eat here again",
            "Poor value for money",
            "Terrible portion sizes",
            "Unpleasant atmosphere",
            "Food was tasteless",
            "Slow and inattentive service",
            "Health concerns after eating here",
        ] * 10

        # Combine all reviews
        texts = (
            positive_product
            + negative_product
            + positive_service
            + negative_service
            + positive_restaurant
            + negative_restaurant
        )

        # Create labels (1 for positive, 0 for negative)
        labels = np.array(
            [1] * len(positive_product)
            + [0] * len(negative_product)
            + [1] * len(positive_service)
            + [0] * len(negative_service)
            + [1] * len(positive_restaurant)
            + [0] * len(negative_restaurant)
        )

        # Shuffle the data
        indices = np.arange(len(texts))
        np.random.seed(42)
        np.random.shuffle(indices)

        return [texts[i] for i in indices], labels[indices]

    def clean_text(self, text):
        """Enhanced text cleaning"""
        import re

        text = str(text).lower()
        # Remove special characters but keep important punctuation
        text = re.sub(r"[^a-zA-Z\s!?.,]", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def prepare_data(self, texts):
        print("Cleaning and tokenizing texts...")
        cleaned_texts = [self.clean_text(text) for text in texts]

        if not self.tokenizer.word_index:
            self.tokenizer.fit_on_texts(cleaned_texts)

        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return padded_sequences

    def build_model(self):
        print("Building model...")
        with tf.device("/CPU:0"):
            model = tf.keras.Sequential(
                [
                    Input(shape=(self.max_len,)),
                    tf.keras.layers.Embedding(self.max_words, 64),
                    # First Conv1D block
                    Conv1D(64, 5, activation="relu", padding="same"),
                    tf.keras.layers.MaxPooling1D(pool_size=2),
                    # Second Conv1D block
                    Conv1D(128, 5, activation="relu", padding="same"),
                    tf.keras.layers.MaxPooling1D(pool_size=2),
                    # Third Conv1D block
                    Conv1D(128, 5, activation="relu", padding="same"),
                    GlobalMaxPooling1D(),
                    # Dense layers
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(64, activation="relu"),
                    Dropout(0.3),
                    Dense(1, activation="sigmoid"),
                ]
            )

            # Use Adam optimizer instead of SGD
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )

            self.model = model
            return model

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
        if self.model is None:
            self.model = self.build_model()

        print(f"\nTraining with {len(x_train)} samples...")
        print(f"Validating with {len(x_val)} samples...")

        try:
            with tf.device("/CPU:0"):
                history = self.model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                )
            return history
        except Exception as e:
            print(f"Error during training: {e}")
            return None

    def predict(self, texts):
        try:
            with tf.device("/CPU:0"):
                X = self.prepare_data(texts)
                predictions = self.model.predict(X, batch_size=32)
                return [
                    "negative" if pred < 0.5 else "positive" for pred in predictions
                ]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return ["error"] * len(texts)


def main():
    try:
        analyzer = SentimentAnalyzer()
        texts, labels = analyzer.get_sample_data()
        X = analyzer.prepare_data(texts)

        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        with tf.device("/CPU:0"):
            history = analyzer.train(X_train, y_train, X_val, y_val)

        if history is not None:
            # Test predictions
            test_texts = [
                "This product is absolutely amazing, exceeded all my expectations!",
                "Terrible service, rude staff and long waiting times",
                "Average experience, nothing special but nothing terrible",
                "Best purchase I've made this year, highly recommend!",
                "Complete waste of money, don't buy this product",
            ]

            print("\nSample Predictions:")
            predictions = analyzer.predict(test_texts)

            for text, pred in zip(test_texts, predictions):
                print(f"\nText: {text}")
                print(f"Predicted sentiment: {pred}")

            # Print prediction probabilities
            with tf.device("/CPU:0"):
                probs = analyzer.model.predict(analyzer.prepare_data(test_texts))
            print("\nPrediction probabilities (probability of positive):")
            for text, prob in zip(test_texts, probs):
                print(f"\nText: {text}")
                print(f"Probability: {prob[0]:.4f}")

            history_dict = {
                "accuracy": [float(x) for x in history.history["accuracy"]],
                "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
                "loss": [float(x) for x in history.history["loss"]],
                "val_loss": [float(x) for x in history.history["val_loss"]],
            }

            with open("training_history.json", "w") as f:
                json.dump(history_dict, f)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
