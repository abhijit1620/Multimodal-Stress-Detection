
🔧 Setup Instructions

1. Clone Repository

```bash
git clone https://github.com/abhijit1620/multimodal-stress-detection.git
cd multimodal-stress-detection


Create Virtual Environment

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows


Install Dependencies

pip install -r requirements.txt


Train Model

python -m src.train        # Train individual models
python -m src.train_fusion # Train fusion model


🚀 Run Streamlit App

streamlit run app/streamlit_app.py

👨‍💻 Authors
•	Abhijeet Sharma – Final Year B.Tech (CS-AIML) 