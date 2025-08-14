from flask import Flask, request, send_file, jsonify
import re  # Correct import for regular expressions
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageDraw
import pytesseract
from PyPDF2 import PdfReader
from model import PIIDetector
from typing import List, Dict

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf', 'txt'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector
try:
    detector = PIIDetector(
        onnx_model_path="id_masker.onnx",
        pt_model_path="best.pt"  # Switch to "last.pt" if needed
    )
except Exception as e:
    print(f"Error loading models: {e}")
    detector = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return text

def extract_text_from_image(image_path):
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def mask_image(image: Image.Image, entities: List[Dict]) -> Image.Image:
    """Mask PII in image by drawing black rectangles over bounding boxes"""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    for entity in entities:
        bbox = entity.get('bbox', [])
        if not bbox or len(bbox) != 4:
            print(f"Skipping invalid bbox (malformed): {bbox}")
            continue
        x1, y1, x2, y2 = bbox
        # Validate bounding box
        x1, x2 = min(x1, x2), max(x1, x2)  # Ensure x2 >= x1
        y1, y2 = min(y1, y2), max(y1, y2)  # Ensure y2 >= y1
        x1, y1 = max(0, x1), max(0, y1)  # Clamp to image bounds
        x2, y2 = min(img_width - 1, x2), min(img_height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping invalid bbox after validation: [{x1}, {y1}, {x2}, {y2}]")
            continue
        draw.rectangle([x1, y1, x2, y2], fill="black")
    return image

@app.route('/health')
def health_check():
    """Endpoint for health checks"""
    return jsonify({"status": "healthy"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text and prepare input
        original_text = ""
        is_image = filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        image = None
        download_link = None
        masked_text = None
        entities = []
        if is_image:
            original_text = extract_text_from_image(filepath)
            image = Image.open(filepath).convert('RGB')
        elif filename.lower().endswith('.pdf'):
            original_text = extract_text_from_pdf(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_text = f.read()

        # Mask text or detect PII
        masked_text = original_text
        if detector:
            entities = detector.detect(image if is_image else original_text, is_image=is_image)
            if is_image:
                # Always save a masked image (even if no detections)
                masked_image = mask_image(image.copy(), entities)
                output_filename = f"masked_{os.path.splitext(filename)[0]}.png"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                masked_image.save(output_path)
                download_link = f"/download/{output_filename}"
            else:
                # For text, mask PII
                entities.sort(key=lambda x: -x['start'])
                for entity in entities:
                    masked_text = masked_text[:entity['start']] + f"[MASKED_{entity['type']}]" + masked_text[entity['end']:]
                # Save masked text
                output_filename = f"masked_{os.path.splitext(filename)[0]}.txt"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(masked_text)
                download_link = f"/download/{output_filename}"
        else:
            # Fallback regex masking for text
            regex_patterns = [
                (r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
                (r'(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}', '[PHONE]'),
                (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
                (r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '[NAME]')
            ]
            for pattern, repl in regex_patterns:  # Renamed 'replacement' to 'repl' for clarity
                masked_text = re.sub(pattern, repl, masked_text, flags=re.IGNORECASE)
            # Save masked text
            output_filename = f"masked_{os.path.splitext(filename)[0]}.txt"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(masked_text)
            download_link = f"/download/{output_filename}"

        return jsonify({
            "status": "success",
            "original_text": original_text,
            "masked_text": masked_text if not is_image else None,
            "entities": entities if is_image else None,
            "download_link": download_link
        })

    except Exception as e:
        print(f"Upload error: {e}")  # Debug logging
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        mimetype = 'image/png' if filename.endswith('.png') else 'text/plain'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            print(f"Download error: File {file_path} not found")  # Debug logging
            return jsonify({"error": "File not found"}), 404
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
    except Exception as e:
        print(f"Download error: {e}")  # Debug logging
        return jsonify({"error": str(e)}), 500