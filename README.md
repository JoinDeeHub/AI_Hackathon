# AI_𝐇𝐚𝐜𝐤𝐚𝐭𝐡𝐨𝐧
𝐅𝐢𝐧𝐚𝐥 𝐏𝐫𝐨𝐣𝐞𝐜𝐭 for 𝐀𝐈 𝐇𝐚𝐜𝐤𝐚𝐭𝐡𝐨𝐧
# ♻️ ReStyleAI – Circular Fashion Platform

**ReStyleAI** is an AI-powered platform that promotes circular fashion by helping users match their existing clothes with sustainable alternatives. It aims to reduce textile waste, support ethical production, and raise awareness about fashion's environmental footprint.

## 🌍 Why ReStyleAI?

> "Every second, one garbage truck full of clothes is burned or buried."

The fashion industry is one of the largest contributors to global pollution. ReStyleAI empowers users to make sustainable choices by intelligently recommending matching items based on uploaded clothing, material type, and weight — using AI/ML-powered matching and impact calculations.

---

## 🚀 Features

- 📸 **Upload Clothing Items** – Drag & drop interface to submit your garment image.
- 🧵 **Material & Weight Selection** – Choose fabric type and weight for accurate impact analysis.
- ♻️ **AI-Based Matching Engine** – Get top sustainable alternatives based on image similarity and eco-impact.
- 🌱 **Environmental Impact Calculator** – See how much CO₂ and water you save.
- 💡 **Circular Economy Score** – Evaluate how eco-friendly your choices are.
- 📊 **Real-World Impact Equivalents** – Understand sustainability through practical analogies.

---

## 🧠 How It Works

1. Upload an image of a clothing item (JPG, PNG, JPEG – up to 200MB)
2. Select the material type and enter item weight
3. ReStyleAI:
   - Processes image & metadata
   - Queries fashion database using AI
   - Returns best sustainable matches
   - Calculates CO₂ saved, water saved, and circularity score
4. Results displayed with badges and insights

---

## 🛠️ Tech Stack

| Area                     | Technology                        |
|--------------------------|------------------------------------|
| Frontend                 | Streamlit                          |
| Styling                  | Custom CSS                         |
| Image Processing         | PIL, Base64 Encoding               |
| Backend/API Integration  | Python (FastAPI or Flask)          |
| Matching Engine (Mocked) | Image URL + Material + Weight Match|
| Impact Calculation       | Custom logic based on material DB  |

---

## 📁 Project Structure

ReStyleAI/
├── backend/
│   ├── __init__.py      # NEW - empty file
│   ├── main.py
│   ├── ai_engine.py
│   └── firebase_service.py
├── frontend/
│   ├── __init__.py      # NEW - empty file
│   ├── app.py
│   └── utils.py
├── scripts/
│   ├── __init__.py      # NEW - empty file
│   └── deploy_gcp.sh
├── .env
├── docker-compose.yml
├── Dockerfile
└── requirements.txt

Sample UI Preview

<img width="954" height="661" alt="image" src="https://github.com/user-attachments/assets/f73acf96-f486-4db6-8350-ee2fcd5c1200" />
<img width="949" height="621" alt="image" src="https://github.com/user-attachments/assets/d7d1ab86-e195-43b6-8f4f-2d405b7a4581" />

### ⚙️ Setup

```bash
git clone https://github.com/yourusername/ReStyleAI.git
cd ReStyleAI
pip install -r requirements.txt
streamlit run frontend/app.py

Acknowledgements:
Hope AI – AI Hackathon 2025
Unsplash – Sample fashion item images
Streamlit – Rapid prototyping framework

📃 License
This project is developed for hackathon purposes.
© 2025 Deepika Narendran | ReStyleAI – AI Hackathon Submission


📬 Contact
For queries or collaboration:
Deepika Narendran
📧 deepika2.ytb@gmail.com
🔗 LinkedIn - https://www.linkedin.com/in/deepika-narendran/





