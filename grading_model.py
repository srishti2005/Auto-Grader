# grading_model.py
import pytesseract
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, TrOCRProcessor
from PIL import Image
import torch
import cv2
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from spellchecker import SpellChecker

# Download necessary resources
nltk.download('punkt')

# Initialize spell checker
spell = SpellChecker()

# Set Tesseract command (adjust this path according to your system)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    print(f"Warning: Could not set Tesseract path - {str(e)}")

# Load TrOCR model and processor
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    print(f"Error loading TrOCR model: {str(e)}")
    raise

# ========== OCR & CLEANING ==========

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read image file")
            
        if image.shape[0] > 2000 or image.shape[1] > 2000:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        image = cv2.adaptiveThreshold(image, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 15)
        return image
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        raise

def tesseract_ocr(preprocessed_img):
    try:
        config = "--psm 6"
        text = pytesseract.image_to_string(preprocessed_img, config=config)
        return text.strip()
    except Exception as e:
        print(f"Error in Tesseract OCR: {str(e)}")
        return ""

def trocr_ocr(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as e:
        print(f"Error in TrOCR processing: {str(e)}")
        return ""

def clean_and_correct(text):
    try:
        from nltk.tokenize import sent_tokenize
        cleaned = re.sub(r"[_~—–]+", " ", text)
        cleaned = re.sub(r"[^\w\s.]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        sentences = sent_tokenize(cleaned)

        corrected_sentences = []
        for sentence in sentences:
            words = sentence.lower().split()
            corrected_words = []
            for word in words:
                if len(word) <= 2 or word in spell:
                    corrected_words.append(word)
                else:
                    corrected = spell.correction(word)
                    corrected_words.append(corrected if corrected else word)
            corrected_sentence = " ".join(corrected_words)
            corrected_sentences.append(corrected_sentence.capitalize())

        return "\n".join(corrected_sentences)
    except Exception as e:
        print(f"Error in text cleaning: {str(e)}")
        return text

def extract_text(image_path):
    try:
        preprocessed = preprocess_image(image_path)
        text = tesseract_ocr(preprocessed)

        if len(text.strip()) < 20:
            text = trocr_ocr(image_path)

        final_text = clean_and_correct(text)
        return final_text
    except Exception as e:
        print(f"Error in text extraction: {str(e)}")
        return ""

# ========== SCORING & GRADING ==========

def calculate_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return similarity_score
    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        return 0.0

def assess_handwriting_quality(image_path):
    try:
        # Placeholder: Replace with a real model or logic for handwriting quality
        # For now, returning a random float between 0.6 and 1.0
        return round(np.random.uniform(0.6, 1.0), 2)
    except Exception as e:
        print(f"Error in handwriting assessment: {str(e)}")
        return 0.7

def calculate_grade(similarity_score, handwriting_score):
    try:
        final_score = (similarity_score * 0.8) + (handwriting_score * 0.2)
        if final_score >= 0.9:
            return "A (Excellent)"
        elif final_score >= 0.75:
            return "B (Good)"
        elif final_score >= 0.6:
            return "C (Average)"
        elif final_score >= 0.4:
            return "D (Below Average)"
        else:
            return "F (Fail)"
    except Exception as e:
        print(f"Error in grade calculation: {str(e)}")
        return "N/A"