# 🧠 Mental Health Classification Using GloVe Embeddings and GRU Networks

## 📌 Project Overview

This project is a Natural Language Processing (NLP) application that classifies mental health conditions from textual data using Deep Learning. The model leverages **GloVe word embeddings** for semantic text representation and a **Gated Recurrent Unit (GRU)** neural network to capture contextual information in user-written text. A user-friendly **Streamlit** web application allows users to enter text and receive real-time mental health predictions.

## 🚀 Features

* Text preprocessing and cleaning
* Tokenization and sequence padding
* Pre-trained GloVe word embeddings
* GRU-based deep learning model using TensorFlow/Keras
* Real-time prediction through a Streamlit web interface
* Simple and interactive user experience

## 🛠️ Technologies Used

* Python
* Natural Language Processing (NLP)
* TensorFlow & Keras
* GloVe Word Embeddings
* GRU (Gated Recurrent Unit)
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## 📂 Project Workflow

1. Load and preprocess the mental health text dataset.
2. Clean and tokenize the text.
3. Convert words into GloVe embedding vectors.
4. Train a GRU-based deep learning model.
5. Save the trained model.
6. Deploy the model using Streamlit for real-time predictions.

## 🎯 Objective

The objective of this project is to demonstrate how Natural Language Processing and Deep Learning can be combined to automatically analyze textual data and classify mental health-related content. This system provides a practical example of AI-assisted text analysis through an easy-to-use web application.

## 📁 Project Structure

```text
mental-health-nlp/
│── app.py
│── preprocessing.py
│── train_glove_gru.py
│── requirements.txt
│── README.md
│── data/
│── glove/
│── saved_model/
```

## ▶️ Installation

```bash
git clone <repository-url>
cd mental-health-nlp
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

## 📌 Future Improvements

* Support for multiple mental health categories
* Transformer-based models such as BERT or RoBERTa
* Improved preprocessing and hyperparameter tuning
* Explainable AI (XAI) for prediction interpretation
* Cloud deployment for public access

## 👩‍💻 Author

Developed as an NLP and Deep Learning project to demonstrate text classification, sequence modeling, and web deployment using modern AI technologies.
