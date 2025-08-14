from flask import Flask, Response, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageDraw, ImageFilter
from model import PIIDetector
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf', 'txt'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector
try:
    detector = PIIDetector(
        onnx_model_path="id_masker.onnx",
        pt_model_path="best.pt"
    )
except Exception as e:
    print(f"Error loading models: {e}")
    detector = None

# Camera control
camera = None
camera_running = False
camera_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames():
    """Generate frames for live camera feed with PII masking"""
    global camera, camera_running
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            camera_running = False
            return
    while camera_running:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame")
            break
        if detector:
            entities = detector.detect(frame, is_frame=True)
            frame = detector.mask_frame(frame, entities, mask_style="black_box")  # Default for live feed
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            print("Failed to encode frame")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    camera_running = False

@app.route('/health')
def health_check():
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
    mask_style = request.form.get('mask_style', 'black_box')
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        entities = []
        download_link = None
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(filepath)
            if detector:
                entities = detector.detect(img, is_image=True)
                masked_img = detector.mask_frame(img, entities, mask_style=mask_style)
                output_filename = f"masked_{os.path.splitext(filename)[0]}.png"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                cv2.imwrite(output_path, masked_img)
                download_link = f"/download/{output_filename}"
            return jsonify({
                "status": "success",
                "original_text": "",
                "entities": entities,
                "download_link": download_link
            })
        elif filename.lower().endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if detector:
                entities = detector.detect(text)
                masked_text = text
                entities.sort(key=lambda x: -x['start'])
                for entity in entities:
                    masked_text = masked_text[:entity['start']] + f"[MASKED_{entity['type']}]" + masked_text[entity['end']:]
                output_filename = f"masked_{os.path.splitext(filename)[0]}.txt"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(masked_text)
                download_link = f"/download/{output_filename}"
            return jsonify({
                "status": "success",
                "original_text": text,
                "masked_text": masked_text,
                "entities": entities,
                "download_link": download_link
            })
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        mimetype = 'image/png' if filename.endswith('.png') else 'text/plain'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            print(f"Download error: File {file_path} not found")
            return jsonify({"error": "File not found"}), 404
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/live')
def live_stream():
    """Stream live camera feed with PII masking"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera detection"""
    global camera, camera_running
    if not camera_running:
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    return jsonify({"error": "Failed to open camera"}), 500
        camera_running = True
        threading.Thread(target=generate_frames, daemon=True).start()
        return jsonify({"status": "Camera started"}), 200
    return jsonify({"status": "Camera already running"}), 200

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera detection"""
    global camera, camera_running
    camera_running = False
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"status": "Camera stopped"}), 200

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture and mask a single frame from the camera"""
    global camera, camera_running
    if not camera_running:
        return jsonify({"error": "Camera not running"}), 400
    mask_style = request.form.get('mask_style', 'black_box')
    try:
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    return jsonify({"error": "Camera not initialized"}), 500
            success, frame = camera.read()
            if not success:
                return jsonify({"error": "Failed to capture frame"}), 500
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"captured_image_{timestamp}.png")
        entities = []
        if detector:
            entities = detector.detect(frame, is_frame=True)
            masked_frame = detector.mask_frame(frame, entities, mask_style=mask_style)
            cv2.imwrite(filepath, masked_frame)
        else:
            cv2.imwrite(filepath, frame)  # Fallback: save unmasked frame
        return jsonify({
            "status": "success",
            "filepath": filepath,
            "download_link": f"/download/{os.path.basename(filepath)}",
            "entities": entities,
            "original_text": ""
        })
    except Exception as e:
        print(f"Capture error: {e}")
        return jsonify({"error": f"Capture error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
