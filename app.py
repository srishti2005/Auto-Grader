import io
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
from functools import wraps
import numpy as np
from flask.json import JSONEncoder
from grading_model import extract_text, calculate_similarity, assess_handwriting_quality, calculate_grade

app = Flask(__name__)

class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = NumpyJSONEncoder

app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
DATABASE = 'users.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                           (username, email, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Username or email already exists.")
        except Exception as e:
            return render_template('signup.html', error=f"An error occurred: {str(e)}")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    if 'master_file' not in request.files or not request.files.getlist('student_files'):
        return jsonify({'error': 'Both master and student files are required'}), 400

    master_file = request.files['master_file']
    student_files = request.files.getlist('student_files')

    if master_file.filename == '' or any(f.filename == '' for f in student_files):
        return jsonify({'error': 'Empty filename detected'}), 400

    if master_file and allowed_file(master_file.filename) and \
       all(allowed_file(f.filename) for f in student_files):
        
        master_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(master_file.filename))
        master_file.save(master_path)
        
        results = []
        try:
            master_text = extract_text(master_path)
            if not master_text:
                raise ValueError("Master text extraction failed")

            for student_file in student_files:
                student_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(student_file.filename))
                student_file.save(student_path)
                
                student_text = extract_text(student_path)
                if not student_text:
                    raise ValueError(f"Text extraction failed for {student_file.filename}")

                similarity_score = calculate_similarity(master_text, student_text)
                handwriting_score = assess_handwriting_quality(student_path)
                final_grade = calculate_grade(similarity_score, handwriting_score)

                results.append({
                    'student_filename': student_file.filename,
                    'student_text': student_text,
                    'similarity': float(similarity_score),
                    'hw_quality': float(handwriting_score),
                    'grade': final_grade,
                    'grade_value': float(similarity_score * 10)
                })

                os.remove(student_path)

            session['results'] = {
                'master_filename': master_file.filename,
                'master_text': master_text,
                'students': results
            }
            os.remove(master_path)
            return redirect(url_for('results'))
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if 'master_path' in locals() and os.path.exists(master_path):
                os.remove(master_path)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results')
@login_required
def results():
    if 'results' not in session:
        return redirect(url_for('dashboard'))
    return render_template('results.html', results=session['results'])

@app.route('/download_text')
@login_required
def download_text():
    if 'results' not in session:
        return redirect(url_for('dashboard'))

    # Build text content directly in memory
    text_content = f"Master Answer Sheet ({session['results']['master_filename']}):\n"
    text_content += session['results']['master_text'] + "\n\n"
    
    for student in session['results']['students']:
        text_content += f"Student Answer Sheet ({student['student_filename']}):\n"
        text_content += student['student_text'] + "\n\n"

    # Create in-memory file buffer
    buffer = BytesIO()
    buffer.write(text_content.encode('utf-8'))
    buffer.seek(0)

    # Create download response
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"extracted_text_{timestamp}.txt",
        mimetype='text/plain'
    )
@app.route('/download')
@login_required
def download():
    if 'results' not in session:
        return redirect(url_for('dashboard'))

    data = {
        'Student File': [],
        'Handwriting Quality': [],
        'Text Similarity': [],
        'Final Grade': [],
        'Grade Value': []
    }

    for student in session['results']['students']:
        data['Student File'].append(student['student_filename'])
        data['Handwriting Quality'].append(f"{student['hw_quality']:.2f}/1.0")
        data['Text Similarity'].append(f"{student['similarity']:.2f}")
        data['Final Grade'].append(student['grade'])
        data['Grade Value'].append(student['grade_value'])

    df = pd.DataFrame(data)
    
    # Create in-memory Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    
    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"grading_report_{timestamp}.xlsx"
    
    return send_file(
        output,
        download_name=filename,
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

