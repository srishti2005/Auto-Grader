# 📝 Auto-Grader – ML-Powered Answer Sheet Grading System

Auto-Grader is a machine learning-based web application that automates the grading of handwritten answer sheets. It uses OCR (Tesseract + TrOCR), NLP, and similarity measures to evaluate student responses against a master answer and generates smart grades based on handwriting quality and content similarity.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgreen)
![Deployed](https://img.shields.io/badge/Deployed-Railway-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🚀 Features

- 🔐 User Authentication (Signup/Login)
- 📤 Upload master and student handwritten answer sheets
- 🧠 Text extraction using **Tesseract** and **TrOCR**
- 🧽 Cleaning, correction using **NLTK** and **SpellChecker**
- 📊 Similarity comparison via **TF-IDF + Cosine Similarity**
- ✍️ Handwriting quality score (randomized for now)
- 🏆 Grade calculation and visualization
- 📥 Export results as `.txt` and `.xlsx` files

---

## 🖼️ Demo

Coming Soon... (Add a Railway or video demo link here)

---

## 🧠 Technologies Used

| Category        | Tools/Frameworks                                     |
|----------------|-------------------------------------------------------|
| **Backend**     | Flask, Gunicorn                                       |
| **OCR**         | Tesseract OCR, TrOCR (Hugging Face Transformers)     |
| **ML/NLP**      | NLTK, TF-IDF, Cosine Similarity                       |
| **Frontend**    | HTML, CSS (Jinja2 templates)                          |
| **Database**    | SQLite3                                               |
| **Others**      | Pandas, OpenCV, EasyOCR, Pillow                       |

---

## 🏗️ Project Structure
Auto_Grading/
│
├── app.py # Main Flask backend
├── grading_model.py # OCR and grading logic
├── requirements.txt # Python dependencies
├── Dockerfile # Docker setup for Railway
├── Procfile # Process runner for deployment
├── templates/ # HTML frontend (Jinja2)
├── static/ # CSS & static assets
├── uploads/ # Temporary file uploads
└── users.db # SQLite database


## 🛠️ Setup Instructions (Local)

### 1. Clone the repository

```bash
git clone https://github.com/srishti2005/Auto-Grader.git
cd Auto-Grader
### 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate    # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
### 3. Download NLTK Data
python -m nltk.downloader punkt
### 4. Run the app locally
python app.py

