import os
import re
import time
import json
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)

# Use /tmp for uploads in cloud (read-only filesystem), local dir otherwise
_local_uploads = os.path.join(os.path.dirname(__file__), 'uploads')
try:
    os.makedirs(_local_uploads, exist_ok=True)
    _test = os.path.join(_local_uploads, '.write_test')
    open(_test, 'w').close()
    os.remove(_test)
    UPLOAD_DIR = _local_uploads
except OSError:
    UPLOAD_DIR = os.path.join('/tmp', 'ocr_uploads')
    os.makedirs(UPLOAD_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Configure Gemini
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = 'gemini-2.5-flash'

# Rate limiter for Gemini free tier (15 RPM)
_gemini_lock = threading.Lock()
_last_call_time = 0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath):
    """Enhance invoice image before sending to Gemini. Returns path to enhanced image."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")

    h, w = img.shape[:2]

    # Upscale small images for better text recognition
    if max(h, w) < 1500:
        scale = 1500 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Light denoise (preserves text while reducing JPEG artifacts)
    img = cv2.fastNlMeansDenoisingColored(img, None, h=6, hForColorComponents=6,
                                           templateWindowSize=7, searchWindowSize=21)

    enhanced_path = filepath + '_enhanced.jpg'
    cv2.imwrite(enhanced_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return enhanced_path


INVOICE_PROMPT = """You are an expert invoice OCR system. Analyze this Indian invoice/bill image carefully.

Extract ALL text and data, including handwritten text in Hindi and English.

Return a JSON object with this exact structure:
{
  "seller": {
    "name": "",
    "address": "",
    "gstin": "",
    "phone": "",
    "email": ""
  },
  "buyer": {
    "name": "",
    "address": "",
    "gstin": "",
    "phone": ""
  },
  "invoice_info": {
    "invoice_number": "",
    "date": "",
    "type": "",
    "vehicle_number": "",
    "transport_name": "",
    "state": "",
    "state_code": ""
  },
  "items": [
    {
      "sno": 1,
      "description": "",
      "hsn_code": "",
      "quantity": "",
      "unit": "",
      "rate": "",
      "amount": ""
    }
  ],
  "subtotal": "",
  "packing_charges": "",
  "cgst": {"rate": "", "amount": ""},
  "sgst": {"rate": "", "amount": ""},
  "igst": {"rate": "", "amount": ""},
  "grand_total": "",
  "amount_in_words": "",
  "bank_details": {
    "bank_name": "",
    "account_number": "",
    "ifsc_code": "",
    "branch": ""
  },
  "raw_text": ""
}

Rules:
- For handwritten Hindi text, transliterate carefully. Include both Devanagari and romanized forms where possible.
- Leave fields as empty string "" if not found in the invoice.
- For amounts, include the number only (no currency symbols).
- The "raw_text" field should contain ALL visible text on the invoice, line by line.
- Be precise with numbers - they are critical for accounting.
- If a field is partially legible, provide your best reading with a note.

Return ONLY valid JSON, no markdown formatting, no code blocks."""


def run_gemini_ocr(filepath):
    """Send image to Gemini Vision API for OCR. Returns (structured_data, elapsed_seconds)."""
    global _last_call_time
    start = time.time()

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Get a free key at https://aistudio.google.com/app/apikey")

    # Preprocess image
    enhanced_path = None
    try:
        enhanced_path = preprocess_image(filepath)
        img = Image.open(enhanced_path)
    except Exception as e:
        print(f"Preprocessing failed, using original: {e}")
        img = Image.open(filepath)

    # Rate limit: at least 4s between calls
    with _gemini_lock:
        now = time.time()
        wait = max(0, 4.0 - (now - _last_call_time))
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()

    # Call Gemini
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        [INVOICE_PROMPT, img],
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=4096,
        ),
    )

    # Parse JSON response
    text = response.text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        structured = json.loads(text)
    except json.JSONDecodeError:
        structured = {
            "seller": {"name": ""},
            "buyer": {"name": ""},
            "invoice_info": {"invoice_number": "", "date": ""},
            "items": [],
            "grand_total": "",
            "raw_text": text
        }

    # Clean up enhanced image
    if enhanced_path and os.path.exists(enhanced_path):
        try:
            os.remove(enhanced_path)
        except OSError:
            pass

    elapsed = round(time.time() - start, 1)
    return structured, elapsed


@app.route('/')
def index():
    project_dir = os.path.dirname(__file__)
    existing_images = [f for f in os.listdir(project_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    existing_images.sort()
    return render_template('index.html', existing_images=existing_images)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                structured, elapsed = run_gemini_ocr(filepath)
                results.append({
                    'filename': file.filename,
                    'image_url': f'/image/{filename}',
                    'structured': structured,
                    'processing_time': elapsed
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'image_url': f'/image/{filename}',
                    'error': str(e),
                    'processing_time': 0
                })

    return jsonify({'results': results})


@app.route('/process-existing', methods=['POST'])
def process_existing():
    data = request.get_json()
    filename = data.get('filename', '')

    project_dir = os.path.dirname(__file__)
    filepath = os.path.join(project_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        structured, elapsed = run_gemini_ocr(filepath)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'results': [{
            'filename': filename,
            'image_url': f'/image/{filename}',
            'structured': structured,
            'processing_time': elapsed
        }]
    })


@app.route('/image/<path:filename>')
def serve_image(filename):
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(upload_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    project_dir = os.path.dirname(__file__)
    return send_from_directory(project_dir, filename)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'gemini_configured': bool(GEMINI_API_KEY),
        'model': GEMINI_MODEL
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
