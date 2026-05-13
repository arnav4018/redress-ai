# ============================================
# FLASK BACKEND — PRODUCTION READY
# Refactored for Randomized Unique Complaint IDs
# ============================================

import os
import re
import pickle
import sqlite3
import nltk
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from translator import detect_and_translate
from sentiment import analyze_sentiment, get_combined_priority
from id_generator import generate_unique_id

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)

# ============================================
# LOAD AI MODEL (train if not exists)
# ============================================

if not os.path.exists('model/classifier.pkl'):
    print("🤖 Model not found. Training now...")
    from startup import train_and_save
    train_and_save()

with open('model/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ AI Model loaded!")

# ============================================
# DATABASE SETUP
# ============================================

def init_db():
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='grievances'")
    table_exists = cursor.fetchone()

    if not table_exists:
        # Create new table with complaint_id column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grievances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                location TEXT NOT NULL,
                complaint TEXT NOT NULL,
                translated_complaint TEXT,
                original_language TEXT,
                department TEXT NOT NULL,
                urgency TEXT NOT NULL,
                confidence REAL NOT NULL,
                sentiment_score REAL,
                sentiment_label TEXT,
                priority_level TEXT,
                status TEXT DEFAULT 'Pending',
                submitted_at TEXT NOT NULL
            )
        ''')
    else:
        # Table exists - check if complaint_id column exists
        cursor.execute("PRAGMA table_info(grievances)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'complaint_id' not in columns:
            print("🔄 Adding complaint_id column to existing database...")
            cursor.execute('ALTER TABLE grievances ADD COLUMN complaint_id TEXT UNIQUE')

    # Add index on complaint_id for fast lookups (critical for tracking)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_complaint_id ON grievances(complaint_id)')

    conn.commit()
    conn.close()
    print("✅ Database ready!")

def migrate_existing_records():
    """Migration: Add complaint_id to existing records if missing"""
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()

    # Check if any records lack complaint_id
    cursor.execute("SELECT id FROM grievances WHERE complaint_id IS NULL OR complaint_id = ''")
    records_needing_id = cursor.fetchall()

    if records_needing_id:
        print(f"🔄 Migrating {len(records_needing_id)} existing records...")
        for record_id in records_needing_id:
            # Generate unique ID for existing record
            new_id = generate_unique_id()
            cursor.execute('UPDATE grievances SET complaint_id = ? WHERE id = ?', (new_id, record_id[0]))

        conn.commit()
        print(f"✅ Migration complete: {len(records_needing_id)} records updated")

    conn.close()

# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ============================================
# URGENCY DETECTION
# ============================================

URGENT_KEYWORDS = [
    'dangerous', 'sparking', 'accident', 'emergency', 'fire',
    'electric shock', 'collapsed', 'dead', 'injury', 'contaminated',
    'burst', 'flooding', 'fallen', 'tilting', 'health', 'hospital',
    'bleeding', 'critical', 'urgent', 'immediate', 'life', 'death',
    'खतरनाक', 'आपातकाल', 'जानलेवा', 'मृत', 'खतरा'
]

def check_urgency(text):
    text_lower = text.lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in text_lower:
            return 'URGENT'
    return 'NORMAL'

# ============================================
# FULL AI PREDICTION
# ============================================

def predict_department(complaint_text):
    translated_text, original_language = detect_and_translate(complaint_text)
    urgency = check_urgency(complaint_text + ' ' + translated_text)
    sentiment_score, sentiment_label, _ = analyze_sentiment(translated_text)
    priority_level = get_combined_priority(urgency, sentiment_label)
    cleaned = clean_text(translated_text)
    vectorized = vectorizer.transform([cleaned])
    department = model.predict(vectorized)[0]
    confidence = round(model.predict_proba(vectorized).max() * 100, 2)
    return (department, confidence, urgency,
            translated_text, original_language,
            sentiment_score, sentiment_label, priority_level)

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM grievances ORDER BY id DESC')
    grievances = cursor.fetchall()
    cursor.execute('SELECT department, COUNT(*) FROM grievances GROUP BY department')
    dept_stats = cursor.fetchall()
    cursor.execute('SELECT urgency, COUNT(*) FROM grievances GROUP BY urgency')
    urgency_stats = cursor.fetchall()
    cursor.execute('SELECT COUNT(*) FROM grievances')
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM grievances WHERE urgency='URGENT'")
    urgent_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM grievances WHERE status='Pending'")
    pending_count = cursor.fetchone()[0]
    cursor.execute('SELECT priority_level, COUNT(*) FROM grievances GROUP BY priority_level')
    priority_stats = cursor.fetchall()
    conn.close()
    return render_template('dashboard.html',
                           grievances=grievances,
                           dept_stats=dept_stats,
                           urgency_stats=urgency_stats,
                           priority_stats=priority_stats,
                           total=total,
                           urgent_count=urgent_count,
                           pending_count=pending_count)

@app.route('/track')
def track():
    return render_template('track.html')

@app.route('/submit', methods=['POST'])
def submit_complaint():
    try:
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        location = request.form.get('location', '').strip()
        complaint = request.form.get('complaint', '').strip()

        if not all([name, phone, location, complaint]):
            return jsonify({'success': False, 'error': 'All fields are required!'})

        if len(complaint) < 5:
            return jsonify({'success': False, 'error': 'Please describe your complaint in more detail.'})

        # Generate unique complaint ID before insertion
        unique_complaint_id = generate_unique_id()

        (department, confidence, urgency,
         translated, language,
         sentiment_score, sentiment_label,
         priority_level) = predict_department(complaint)

        conn = sqlite3.connect('grievances.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO grievances
            (complaint_id, name, phone, location, complaint, translated_complaint,
             original_language, department, urgency, confidence,
             sentiment_score, sentiment_label, priority_level,
             status, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (unique_complaint_id, name, phone, location, complaint, translated,
              language, department, urgency, confidence,
              sentiment_score, sentiment_label, priority_level,
              'Pending', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'complaint_id': unique_complaint_id,
            'department': department,
            'confidence': confidence,
            'urgency': urgency,
            'language': language,
            'sentiment_label': sentiment_label,
            'priority_level': priority_level,
            'translated': translated if language == 'Hindi' else None,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Something went wrong: {str(e)}'})

@app.route('/track_complaint', methods=['POST'])
def track_complaint():
    try:
        complaint_id = request.form.get('complaint_id', '').strip()
        if not complaint_id:
            return jsonify({'success': False, 'error': 'Please enter a complaint ID'})

        conn = sqlite3.connect('grievances.db')
        cursor = conn.cursor()
        # Use the new complaint_id column with indexed lookup
        cursor.execute('SELECT * FROM grievances WHERE complaint_id=?', (complaint_id.upper(),))
        g = cursor.fetchone()
        conn.close()

        if not g:
            return jsonify({'success': False, 'error': f'No complaint found with ID {complaint_id}'})

        # Return data with new complaint_id field (complaint_id is at index 15)
        return jsonify({
            'success': True,
            'complaint_id': g[15],
            'name': g[1], 'location': g[3],
            'complaint': g[4], 'language': g[6],
            'department': g[7], 'urgency': g[8],
            'confidence': g[9], 'sentiment_label': g[11],
            'priority_level': g[12], 'status': g[13],
            'submitted_at': g[14]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_status', methods=['POST'])
def update_status():
    complaint_id = request.form.get('complaint_id')
    new_status = request.form.get('status')
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE grievances SET status=? WHERE complaint_id=?', (new_status, complaint_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('grievances.db')
    cursor = conn.cursor()
    cursor.execute('SELECT department, COUNT(*) FROM grievances GROUP BY department')
    dept_data = dict(cursor.fetchall())
    cursor.execute('SELECT urgency, COUNT(*) FROM grievances GROUP BY urgency')
    urgency_data = dict(cursor.fetchall())
    conn.close()
    return jsonify({'departments': dept_data, 'urgency': urgency_data})

# ============================================
# RUN
# ============================================

init_db()
migrate_existing_records()  # Add complaint_id to any existing records

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)