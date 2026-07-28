🧠 Mental Health Intensity Prediction using NLP
📌 Overview
Mental health issues are becoming increasingly common, making early identification of emotional distress more important than ever. This project presents an NLP-based Mental Health Intensity Prediction System that analyzes text posts and classifies them into different mental health intensity levels using Deep Learning.

The application uses a pre-trained GloVe (Global Vectors for Word Representation) embedding with a GRU (Gated Recurrent Unit) neural network to understand the contextual meaning of text and predict the intensity level. The trained model is deployed through a Flask web application, allowing users to perform both single-text and batch predictions.

Disclaimer: This project is intended for educational and research purposes only. It is not a substitute for professional mental health diagnosis or treatment.

🚀 Features
Predict mental health intensity from user text
Batch prediction using CSV file upload
Text preprocessing pipeline
Pre-trained GloVe word embeddings
GRU-based deep learning model
Interactive Flask web interface
Download prediction results
Responsive and user-friendly dashboard
📂 Project Structure
Mental-Health-Intensity/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── gru_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── utils/
│   ├── predictor.py
│   ├── preprocessing.py
│   └── performance.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── uploads/
├── outputs/
└── notebook/
    └── main.ipynb
🛠️ Technologies Used
Python
Natural Language Processing (NLP)
TensorFlow / Keras
GloVe Word Embeddings
GRU (Gated Recurrent Unit)
Flask
Pandas
NumPy
Scikit-learn
HTML
CSS
JavaScript
📊 Dataset
The dataset contains mental health-related text posts with corresponding intensity labels.

Example columns:

Column	Description
posts	User-generated text
intensity	Mental health intensity label
The model predicts one of the following categories:

Normal
Low
Moderate
High
🔄 Data Preprocessing
The preprocessing pipeline includes:

Convert text to lowercase
Remove URLs
Remove HTML tags
Remove punctuation
Remove special characters
Remove extra whitespace
Tokenization
Sequence padding
🧠 Model Architecture
The model uses:

Pre-trained GloVe Embedding Layer
GRU Layer
Dropout Layer
Dense Layer
Softmax Output Layer
The embedding layer is initialized using pre-trained GloVe vectors to capture semantic relationships between words.

📈 Evaluation Metrics
Model performance is evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report
🌐 Flask Application
The web application supports:

Single Prediction
Users can enter a sentence and receive the predicted mental health intensity.

Batch Prediction
Users can upload a CSV file containing multiple text records.

The application:

Reads the uploaded file
Performs preprocessing
Predicts intensity for each record
Displays results
Allows downloading predictions as CSV
⚙️ Installation
Clone the repository:

git clone https://github.com/your-username/mental-health-intensity.git
Move into the project directory:

cd mental-health-intensity
Create a virtual environment (optional):

python -m venv venv
Activate it:

Windows

venv\Scripts\activate
Linux / Mac

source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Run the Flask application:

python app.py
Open your browser:

http://127.0.0.1:5000
💻 Usage
Launch the Flask application.
Enter a text message or upload a CSV file.
Click Predict.
View the predicted mental health intensity.
Download results for batch predictions.
📷 Application Preview
Add screenshots of your application here.

Example:

screenshots/
    home.png
    prediction.png
    batch_prediction.png
🔮 Future Enhancements
BERT-based model implementation
Explainable AI (XAI)
Real-time API deployment
Multi-language support
User authentication
Dashboard analytics
Cloud deployment
Mobile application integration
👩‍💻 Author
Chanuvala Swetha

B.Tech – Computer Science Engineering

Interested in:

Machine Learning
Natural Language Processing
Deep Learning
AI Applications
Data Science
📄 License
This project is licensed under the MIT License.

⭐ Acknowledgements
TensorFlow
Keras
Scikit-learn
Flask
GloVe Embeddings
NumPy
Pandas
Open-source NLP community