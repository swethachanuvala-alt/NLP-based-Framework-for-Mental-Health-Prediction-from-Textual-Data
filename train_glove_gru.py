"""
train_glove_gru.py
-------------------
Reproduces the GloVe + GRU model from the notebook end-to-end and
saves the three artifacts the app needs:

    saved_model/glove_gru.keras
    saved_model/tokenizer.pkl
    saved_model/label_encoder.pkl

Run this ONCE before using app.py.

Requirements before running:
  1. data/Mental_Health_Dataset.csv   (already included)
  2. glove/glove.6B.100d.txt          (download separately, see README.md)

Usage:
    python train_glove_gru.py
"""

import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import preprocess_batch

DATA_PATH = "data/Mental_Health_Dataset.csv"
GLOVE_PATH = "glove/glove.6B.100d.txt"
MODEL_OUT = "saved_model/glove_gru.keras"
TOKENIZER_OUT = "saved_model/tokenizer.pkl"
LABEL_ENCODER_OUT = "saved_model/label_encoder.pkl"

MAX_LEN = 100          # sequence length used for training (matches notebook cell 13)
EMBEDDING_DIM = 100    # glove.6B.100d
EPOCHS = 10
BATCH_SIZE = 32
RANDOM_STATE = 42


def load_and_clean_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    print(f"  {len(df)} rows after dropping nulls")

    print("Cleaning + tokenizing + lemmatizing text (this takes a bit)...")
    df["posts"] = preprocess_batch(df["posts"])
    return df


def build_embedding_matrix(word_index):
    print("Loading GloVe vectors...")
    embeddings_index = {}
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    print(f"  Loaded {len(embeddings_index)} word vectors.")

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    hits = 0
    for word, i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
            hits += 1
    print(f"  Matched {hits}/{len(word_index)} vocab words to GloVe vectors.")
    return embedding_matrix


def build_model(vocab_size, embedding_matrix, num_classes):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            trainable=False,
        )
    )
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    df = load_and_clean_data()

    X = df["posts"]
    y = df["intensity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Fitting tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("Balancing classes with RandomOverSampler...")
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train_pad, y_train_enc)

    embedding_matrix = build_embedding_matrix(tokenizer.word_index)
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(np.unique(y_train_enc))

    print("Building GloVe + GRU model...")
    model = build_model(vocab_size, embedding_matrix, num_classes)
    model.summary()

    print("Training...")
    model.fit(
        X_train_balanced,
        y_train_balanced,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_pad, y_test_enc),
    )

    loss, accuracy = model.evaluate(X_test_pad, y_test_enc)
    print(f"GloVe + GRU test accuracy: {accuracy:.4f}")

    print("Saving artifacts...")
    model.save(MODEL_OUT)
    with open(TOKENIZER_OUT, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(LABEL_ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)

    print("Done. Saved:")
    print(f"  {MODEL_OUT}")
    print(f"  {TOKENIZER_OUT}")
    print(f"  {LABEL_ENCODER_OUT}")


if __name__ == "__main__":
    main()
