🩺 MediPredict Pro — AI Disease Prediction System

🔗 Live App: https://medipredict-pro-ywd9o8j42t3dvxavzc7uyt.streamlit.app/

MediPredict Pro is an AI-powered healthcare analytics application that predicts probable diseases based on user-selected symptoms using machine learning.

🚀 Features
🔍 Symptom-based disease prediction
🧠 Machine Learning model (Naive Bayes - Scikit-learn)
📊 132 Symptoms • 41 Diseases • 4,920 Clinical Records
⚡ Instant predictions with confidence scoring
🎨 Clean and interactive UI (Streamlit)
📚 Built-in disease catalog with top symptoms
🧠 How It Works
User selects symptoms from the interface
Symptoms are converted into a binary feature vector
Trained ML model predicts the most probable disease
Confidence score is calculated dynamically
🛠 Tech Stack
Python
Streamlit
Pandas
NumPy
Scikit-learn
Joblib
📂 Project Structure
MediPredict/
│── app.py
│── model.pkl
│── Training.csv
│── requirements.txt
│── runtime.txt
│── README.md
▶️ Run Locally
1. Clone the repository
git clone https://github.com/macwinroym/medipredict-pro.git
cd medipredict-pro
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app.py
🌐 Deployment

This project is deployed using Streamlit Cloud and accessible via the live link above.

⚠️ Disclaimer

This application is intended for educational and demonstration purposes only.
It does not provide medical diagnosis and should not replace consultation with a qualified healthcare professional.

🚀 Future Improvements
Probability-based prediction visualization
Doctor recommendation system
User login & patient history tracking
Full-stack upgrade (React + FastAPI)
Mobile-friendly UI
👨‍💻 Author

Macwin Roy M
Final Year Project — AI / Machine Learning
