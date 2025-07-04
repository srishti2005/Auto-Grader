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

