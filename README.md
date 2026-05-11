# 🇮🇳 Redress AI — AI-Based Citizen Grievance Classification System

> Built for BGI Hackathon 2026 | Team: [Your Team Name]

## 🚀 Live Demo
🔗 [Click here to view live demo](#) ← (we'll add this after deployment)

---

## 🧠 What is Redress AI?
Redress AI is an intelligent grievance processing system that automatically
classifies citizen complaints into relevant government departments using
NLP and Machine Learning.

## ✨ Features
- 🤖 **93% AI Accuracy** — TF-IDF + Logistic Regression classifier
- 🌐 **Multilingual** — English & Hindi support with auto-translation
- 🚨 **Smart Urgency Detection** — keyword-based priority flagging
- 😟 **Sentiment Analysis** — TextBlob-powered complaint severity scoring
- 📊 **Admin Dashboard** — real-time charts and complaint management
- 🔍 **Complaint Tracking** — citizens can track status by ID

## 🏢 Departments Supported
| Department | Example Complaint |
|---|---|
| 💧 Water Supply | No water since 3 days |
| ⚡ Electricity | Sparking wire near house |
| 🛣️ Roads | Big potholes on main road |
| 🗑️ Sanitation | Garbage not collected |
| 🏛️ Public Services | Pension not received |

## 🛠️ Tech Stack
- **Backend** — Python, Flask
- **AI/ML** — scikit-learn, NLTK, TextBlob
- **Translation** — deep-translator, langdetect
- **Frontend** — HTML, CSS, JavaScript, Chart.js
- **Database** — SQLite

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/redress-ai.git
cd redress-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the AI model
```bash
python model/train_model.py
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser

## 📁 Project Structure
redress-ai/
├── model/
│   └── train_model.py    ← AI training script
├── templates/
│   ├── index.html        ← Complaint form
│   ├── dashboard.html    ← Admin dashboard
│   └── track.html        ← Complaint tracker
├── static/
│   └── style.css
├── app.py                ← Flask backend
├── translator.py         ← Hindi translation
├── sentiment.py          ← Sentiment analysis
└── requirements.txt

## 👥 Team
Built with ❤️ for BGI Hackathon 2026