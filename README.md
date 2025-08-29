
# Clone repo
git clone https://github.com/abhijit1620/multimodal-stress-detection.git
cd multimodal-stress-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt

🧪 Train Model

python -m src.train

🚀 Run Streamlit App

streamlit run app/streamlit_app.py

👨‍💻 Authors
•	Abhijeet Sharma – Final Year B.Tech (CS-AIML)