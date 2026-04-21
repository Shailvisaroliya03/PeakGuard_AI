PeakGuard AI is a state-of-the-art energy forecasting dashboard designed for commercial buildings in India. It uses Machine Learning (LightGBM) to predict electricity demand in real-time, detect contract breaches before they happen, and automate mitigation strategies (Battery Dispatch & HVAC Optimization) to prevent expensive demand penalties.

🚀 Key Features
🧠 1. AI-Powered Forecasting
Uses LightGBM trained on ASHRAE building data to predict load 1 hour in advance.
Factors in Weather (Temp/Cloud), Time-of-Use (TOU) tariffs, and Building Metadata.
🛡️ 2. Interactive Mitigation (Peak Shaving)
Manual Mode: Operators can click to dispatch Battery Storage (50kW) or Optimize HVAC (15% reduction) to fix breaches instantly.
🤖 Auto-Pilot Mode: Autonomous agent detects breaches and executes countermeasures in milliseconds without human intervention.
💰 3. Financial Intelligence (Indian Context)
Real-time tracking of Rupees Saved (₹) vs. Grid Penalties.
Adapts to Dynamic TOU Tariffs (Peak Rates vs. Off-Peak Rates).
🌱 4. Sustainability Tracking
Quantifies Scope 2 Emissions (Grid Carbon) vs. Avoided CO₂ via Solar/Battery usage.
🛠️ Tech Stack
Frontend: Streamlit (Custom CSS, Glassmorphism UI)
Backend: Python, Pandas, NumPy
ML Engine: LightGBM Regressor
Visualization: Altair (Interactive Charts)
📦 Installation
Clone the repo: bash git clone https://github.com/Priyanshuturakhia/PeakGuardAI.git cd PeakGuard-AI

Install dependencies: Bash pip install -r requirements.txt

Run the dashboard: Bash streamlit run app.py
