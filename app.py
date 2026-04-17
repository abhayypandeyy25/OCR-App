import os
import re
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import easyocr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Initialize EasyOCR reader (Hindi + English)
reader = easyocr.Reader(['hi', 'en'], gpu=False)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def deskew(image):
    """Detect and correct skew angle in the image."""
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 100:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(filepath):
    """Preprocess invoice image for better OCR. Returns (binary, enhanced_gray)."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")

    # Step 1: Upscale small images
    h, w = img.shape[:2]
    if w < 2000:
        scale = 2000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Step 2: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Denoise (removes JPEG compression artifacts)
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7,
                                     searchWindowSize=21)

    # Step 4: CLAHE contrast enhancement (critical for colored paper)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 5: Deskew
    enhanced = deskew(enhanced)

    # Step 6: Adaptive threshold binarization
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)

    # Step 7: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, enhanced


def filter_noise(ocr_results, min_confidence=0.15):
    """Remove very low confidence short text (noise)."""
    filtered = []
    for bbox, text, conf in ocr_results:
        text = text.strip()
        if not text:
            continue
        if conf < min_confidence and len(text) < 3:
            continue
        filtered.append((bbox, text, conf))
    return filtered


def run_ocr(filepath):
    """Run OCR with preprocessing and tuned parameters. Returns (results, elapsed_seconds)."""
    start = time.time()

    ocr_params = dict(
        batch_size=1,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
        canvas_size=2560,
        mag_ratio=1.5,
        slope_ths=0.3,
        width_ths=1.0,
        add_margin=0.15,
        paragraph=False,
        min_size=10,
        contrast_ths=0.05,
        adjust_contrast=0.7,
    )

    try:
        binary, enhanced = preprocess_image(filepath)
        # Use enhanced grayscale (CLAHE) — better for colored paper invoices
        best = reader.readtext(enhanced, **ocr_params)
    except Exception as e:
        print(f"Preprocessing failed, falling back to raw OCR: {e}")
        best = reader.readtext(filepath, **ocr_params)

    best = filter_noise(best)
    elapsed = round(time.time() - start, 1)
    return best, elapsed


def extract_structured_data(ocr_results):
    """Parse OCR results into structured invoice data."""
    lines = []
    for bbox, text, confidence in ocr_results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        box_height = abs(bbox[2][1] - bbox[0][1])
        lines.append({
            'text': text.strip(),
            'confidence': round(confidence * 100, 1),
            'y': y_center,
            'x': x_center,
            'bbox': bbox,
            'height': box_height
        })

    if not lines:
        return {'raw_lines': [], 'header': [], 'items': [], 'totals': [], 'metadata': {}}

    # Dynamic row-grouping threshold based on average text height
    avg_height = sum(l['height'] for l in lines) / len(lines)
    row_threshold = max(avg_height * 0.6, 10)

    # Sort by vertical position (top to bottom), then horizontal (left to right)
    lines.sort(key=lambda l: (round(l['y'] / row_threshold) * row_threshold, l['x']))

    # Group lines by approximate Y position (same row)
    rows = []
    current_row = []
    last_y = None
    for line in lines:
        if last_y is None or abs(line['y'] - last_y) < row_threshold:
            current_row.append(line)
        else:
            if current_row:
                rows.append(current_row)
            current_row = [line]
        last_y = line['y']
    if current_row:
        rows.append(current_row)

    # Build structured output
    structured = {
        'raw_lines': [],
        'header': [],
        'items': [],
        'totals': [],
        'metadata': {}
    }

    # Extract key fields using pattern matching
    full_text = ' '.join([l['text'] for l in lines])

    # GSTIN
    gstin_match = re.search(
        r'GST(?:IN)?[:\s.]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z][A-Z][0-9A-Z])',
        full_text, re.IGNORECASE
    )
    if not gstin_match:
        gstin_match = re.search(r'GSTIN[:\s.]*([A-Z0-9]{15})', full_text, re.IGNORECASE)
    if gstin_match:
        structured['metadata']['gstin'] = gstin_match.group(1)

    # Date (English + Hindi labels)
    date_match = re.search(
        r'(?:Date|दिनांक|तारीख|Dt\.?)[:\s.]*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})',
        full_text, re.IGNORECASE
    )
    if date_match:
        structured['metadata']['date'] = date_match.group(1)

    # Invoice/Bill/Serial number
    invoice_match = re.search(
        r'(?:Invoice|Serial|Bill|बिल|चालान)\s*(?:No\.?|Number|नं\.?)?[:\s.]*([A-Z0-9\-/]+\d+)',
        full_text, re.IGNORECASE
    )
    if invoice_match:
        structured['metadata']['invoice_number'] = invoice_match.group(1)

    # Total amount
    total_match = re.search(
        r'(?:Grand\s*Total|Total\s*Amount|Invoice\s*Total|कुल\s*राशि|कुल\s*योग)[:\s]*[₹Rs.]*\s*([\d,]+\.?\d*)',
        full_text, re.IGNORECASE
    )
    if total_match:
        structured['metadata']['total_amount'] = total_match.group(1)

    # Mobile number
    mobile_match = re.search(r'(?:Mob|Mobile|Phone|Tel)[:\s.]*(\d[\d\s\-]{8,})', full_text, re.IGNORECASE)
    if mobile_match:
        structured['metadata']['mobile'] = mobile_match.group(1).strip()

    # Seller/business name: largest text in the top 20% of document
    if lines:
        max_y = max(l['y'] for l in lines)
        top_lines = [l for l in lines if l['y'] < max_y * 0.2]
        if top_lines:
            tallest = max(top_lines, key=lambda l: l['height'])
            if tallest['height'] > avg_height * 1.3:
                structured['metadata']['seller_name'] = tallest['text']

    # Classification keywords
    total_keywords = ['total', 'grand', 'amount', 'gst', 'cgst', 'sgst', 'igst', 'tax',
                      'कुल', 'योग', 'कर', 'राशि', 'packing']
    header_keywords = ['s.no', 's.n', 'description', 'product', 'particulars', 'hsn',
                       'qty', 'quantity', 'rate', 'amount', 'विवरण', 'मात्रा', 'दर']
    footer_keywords = ['terms', 'condition', 'bank', 'signature', 'certified',
                       'e.&o.e', 'e.o.e', 'e.&.o.e', 'शर्तें', 'हस्ताक्षर',
                       'declaration', 'subject to', 'customer sign']

    # Organize rows into sections
    header_done = False
    for row in rows:
        row_texts = [item['text'] for item in row]
        row_text = ' | '.join(row_texts)
        avg_confidence = sum(item['confidence'] for item in row) / len(row)

        row_entry = {
            'text': row_text,
            'cells': row_texts,
            'confidence': round(avg_confidence, 1)
        }

        lower_text = row_text.lower()
        if any(kw in lower_text for kw in total_keywords):
            structured['totals'].append(row_entry)
        elif not header_done and any(kw in lower_text for kw in header_keywords):
            structured['header'].append(row_entry)
            header_done = True
        elif header_done and not any(kw in lower_text for kw in footer_keywords):
            if any(c.isdigit() for c in row_text):
                structured['items'].append(row_entry)

        structured['raw_lines'].append(row_entry)

    return structured


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

            ocr_results, elapsed = run_ocr(filepath)
            structured = extract_structured_data(ocr_results)

            results.append({
                'filename': file.filename,
                'structured': structured,
                'processing_time': elapsed
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

    ocr_results, elapsed = run_ocr(filepath)
    structured = extract_structured_data(ocr_results)

    return jsonify({
        'results': [{
            'filename': filename,
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


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5001)
