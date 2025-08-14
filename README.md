PII Masking Tool
The PII Masking Tool is a Python-based application designed to detect and mask Personally Identifiable Information (PII) in images, PDFs, and text files. It uses a YOLO-based machine learning model for detecting PII in images and regular expressions for text-based PII masking. The tool provides a user-friendly graphical interface built with Tkinter, a Flask-based backend for processing files, and supports both image-based detection (masking PII with black rectangles) and text-based masking (replacing PII with placeholders like [MASKED_EMAIL]). The project is ideal for applications requiring data privacy and compliance with regulations like GDPR or HIPAA.
Table of Contents

Features
Project Architecture
Prerequisites
Installation
Usage
API Endpoints
Troubleshooting
Contributing
License
Contact

Features

Image PII Detection: Identifies PII (e.g., emails, phone numbers, SSNs, names) in images (.png, .jpg, .jpeg) using a YOLO model (best.pt or last.pt) and masks detected regions with black rectangles.
Text/PDF PII Masking: Masks PII in text files (.txt) and PDFs using regex-based patterns, replacing sensitive data with placeholders (e.g., [MASKED_EMAIL]).
Graphical User Interface: Built with Tkinter, allowing users to upload files, view original and masked content, and download processed files.
Flask Backend: Provides a RESTful API for file processing, health checks, and downloading masked files.
Supported PII Types:
Email addresses (e.g., test@example.com)
Phone numbers (e.g., 123-456-7890)
Social Security Numbers (e.g., 123-45-6789)
Names (e.g., John Doe)


Robust Error Handling: Validates bounding boxes to prevent errors, handles invalid file types, and provides detailed debug logs.
Cross-Platform: Runs on Windows, with potential for Linux/macOS with minor adjustments (e.g., Tesseract path).

Project Architecture
The project is structured into four main Python files, each serving a distinct purpose:

model.py:

Contains the PIIDetector class for PII detection.
Uses a YOLO model (best.pt or last.pt) for image-based detection and ONNX model (id_masker.onnx) as an alternative.
Implements regex-based detection for text/PDF inputs.
Handles bounding box validation and scaling to ensure accurate PII detection.


app.py:

Flask backend providing API endpoints:
/health: Checks server status.
/upload: Processes uploaded files (images, PDFs, text) and returns masked results.
/download/<filename>: Serves masked files for download.


Integrates with model.py for PII detection and uses Tesseract OCR for text extraction from images/PDFs.
Implements regex fallback for text masking when models are unavailable.


interface.py:

Tkinter-based GUI for user interaction.
Features file upload, processing, result display (original vs. masked content), and download functionality.
Communicates with the Flask backend via HTTP requests.


run.py:

Entry point to launch the application, starting the Flask backend and Tkinter GUI.



Dependencies

Python Libraries: flask, werkzeug, pillow, pytesseract, PyPDF2, onnxruntime, torch, ultralytics, requests, waitress.
External Tools: Tesseract OCR for text extraction.
Model Files: id_masker.onnx, best.pt, last.pt for PII detection in images.

Directory Structure
pii-masking-tool/
├── model.py
├── app.py
├── interface.py
├── run.py
├── id_masker.onnx
├── best.pt
├── last.pt
├── uploads/  (created automatically)

Prerequisites

Python: Version 3.8 or higher.
Tesseract OCR: Required for extracting text from images and PDFs.
Model Files: Pre-trained YOLO models (best.pt, last.pt) and ONNX model (id_masker.onnx).
System: Tested on Windows; Linux/macOS may require Tesseract path adjustments.

Installation

Clone the Repository:
git clone https://github.com/your-username/pii-masking-tool.git
cd pii-masking-tool


Install Dependencies:Install required Python packages:
pip install flask werkzeug pillow pytesseract PyPDF2 onnxruntime torch ultralytics requests waitress


Install Tesseract OCR:

Download and install Tesseract from GitHub.
Ensure Tesseract is installed at C:\Program Files\Tesseract-OCR\tesseract.exe (Windows). If installed elsewhere, update app.py:pytesseract.pytesseract.tesseract_cmd = r'YOUR_ACTUAL_TESSERACT_PATH'




Place Model Files:

Place id_masker.onnx, best.pt, and last.pt in the project root (pii-masking-tool/).
To use last.pt instead of best.pt, update app.py:detector = PIIDetector(onnx_model_path="id_masker.onnx", pt_model_path="last.pt")


Note: Model files are not included in the repository due to size or sensitivity. Contact the repository owner for access or train your own models.


Create .gitignore:To exclude large or sensitive files:
id_masker.onnx
best.pt
last.pt
uploads/



Usage

Run the Application:
python run.py

This launches the Flask backend (on http://127.0.0.1:5000) and the Tkinter GUI.

GUI Instructions:

Select File: Click "Browse" to select a .png, .jpg, .jpeg, .pdf, or .txt file.
Process File: Click "Process File" to upload and process:
Images: PII is detected using the YOLO model, masked with black rectangles, and saved as masked_<filename>.png. Bounding box details (e.g., Type: EMAIL, Score: 0.85, BBox: [100, 150, 200, 180]) are displayed.
Text/PDF: PII is masked with placeholders (e.g., [MASKED_EMAIL]) and saved as masked_<filename>.txt. Original and masked text are displayed.


View Results: The left panel shows original content, and the right panel shows masked content or detection details.
Download: Click "Download Masked File" to save the masked file (.png for images, .txt for text/PDF).
Retry Connection: Use if the backend fails to connect (e.g., shows "Backend connection failed").


Testing with Sample Files:

Image: Create an image with clear PII (e.g., "Email: test@example.com, Phone: 123-456-7890" in Arial, size 20+, white background, black text).
Text: Create a .txt file with PII (e.g., "Name: John Doe, SSN: 123-45-6789").
PDF: Use a PDF containing similar PII text.



API Endpoints
The Flask backend provides the following endpoints:

Health Check (GET /health):

Returns: {"status": "healthy"} if the server is running.
Example:curl http://127.0.0.1:5000/health




Upload File (POST /upload):

Accepts: A file (.png, .jpg, .jpeg, .pdf, .txt) via form-data (file field).
Returns: JSON with:
status: "success" or "error".
original_text: Extracted text from the file.
masked_text: Masked text (for text/PDF) or null (for images).
entities: List of detected PII with bounding boxes (for images) or null (for text/PDF).
download_link: URL to download the masked file (e.g., /download/masked_image.png).


Example:curl -X POST -F "file=@test_image.png" http://127.0.0.1:5000/upload




Download File (GET /download/<filename>):

Returns: The masked file (.png for images, .txt for text/PDF).
Example:curl http://127.0.0.1:5000/download/masked_test_image.png -o masked_test_image.png





Troubleshooting
1. Circular Import Error

Error: ImportError: cannot import name 'PIIDetector' from partially initialized module 'model'.
Cause: model.py importing from app.py or related modules, creating a circular dependency.
Fix: The provided model.py is self-contained. Ensure no imports like from app import ... exist in model.py.
Verification:python app.py

If the error persists, share the full model.py to identify hidden imports.

2. API Error: x1 must be greater than or equal to xo

Cause: Invalid bounding box coordinates (x2 < x1 or y2 < y1) from the YOLO model.
Fix: The code validates bounding boxes in model.py (_parse_yolo_results) and app.py (mask_image), skipping invalid boxes. Check console logs for:
Skipping invalid box: ...
Skipping invalid bbox (malformed): ...
Raw detections - Boxes: ...


Debugging:
Share the raw detection output and test image resolution (e.g., image.size).
Test with a 640x640 image to isolate scaling issues.



3. No Detections

Logs: Show "0: 448x640 (no detections)" instead of 640x640.
Fix:
Check Model Classes:from ultralytics import YOLO
model = YOLO("best.pt")
print(model.names)  # e.g., {0: 'email', 1: 'phone', ...}

Update model.py class mapping:pii_type = {0: "email", 1: "phone", 2: "ssn", 3: "name"}.get(int(label), "UNKNOWN")


Test Image Quality: Use a clear image with PII (e.g., Arial 24pt, black text on white background).
Lower Confidence Threshold: The code uses conf=0.01. Try conf=0.001:results = self.pt_model.predict(image, imgsz=(640, 640), conf=0.001, verbose=True)


Fix Input Size: If logs show 448x640, re-export the model:yolo export model=best.pt format=onnx imgsz=640,640

Update app.py to use last.pt if needed.



4. Masking and Download Issues

Verification:
Check uploads/ for masked_<filename>.png or masked_<filename>.txt.
Open the file to confirm masking (black rectangles for images, [MASKED_<TYPE>] for text).
Ensure the download button is enabled after processing.


Debugging:
Check console for API response: ... to verify download_link.
Look for Upload error: ... or Download error: ....
Verify file permissions in uploads/.



5. Tesseract Errors

Error: Tesseract not found or incorrect path.
Fix: Update the Tesseract path in app.py or install Tesseract at C:\Program Files\Tesseract-OCR\tesseract.exe.

6. Performance Issues

Cause: Large images or CPU-based processing.
Fix: Resize images to 640x640 before uploading or enable GPU acceleration if available.

Debugging Tips

Console Logs: Monitor for:
PyTorch input image size: ...
Raw detections - Boxes: ...
Detected entities: ...
Skipping invalid box: ... or Skipping invalid bbox (malformed): ...


Test Image: Use a clear image with PII (e.g., "Email: test@example.com" in Arial 24pt).
Model Inspection:
Check model.names for best.pt to verify class mappings.
Use Netron to inspect id_masker.onnx input/output formats (e.g., [1, 3, 640, 640]).



Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please include tests and update documentation as needed.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For issues, questions, or model file access, open an issue on GitHub or contact [your-email@example.com].

cd "C:\project ai\new yolo"
git init
git add model.py app.py interface.py run.py README.md
git commit -m "Initial commit with PII Masking Tool"
git remote add origin https://github.com/your-username/pii-masking-tool.git
git push -u origin main


id_masker.onnx
best.pt
last.pt
uploads/

git add README.md
git commit -m "Add README"
git push
