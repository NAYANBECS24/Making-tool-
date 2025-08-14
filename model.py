import re
import onnxruntime as ort
import torch
from typing import List, Dict
import numpy as np
from PIL import Image
from ultralytics import YOLO

class PIIDetector:
    def __init__(self, onnx_model_path: str = None, pt_model_path: str = None):
        self.ort_session = None
        self.pt_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load ONNX model
        if onnx_model_path:
            try:
                self.ort_session = ort.InferenceSession(onnx_model_path)
                print(f"Successfully loaded ONNX model from {onnx_model_path}")
            except Exception as e:
                print(f"Error loading ONNX model: {e}")

        # Load PyTorch YOLO model
        if pt_model_path:
            try:
                self.pt_model = YOLO(pt_model_path)
                self.pt_model.to(self.device)
                print(f"Successfully loaded PyTorch model from {pt_model_path}")
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")

    def detect(self, input_data, is_image: bool = False) -> List[Dict]:
        """Detect PII entities in text or image"""
        entities = []

        if is_image:
            # Image-based detection
            if self.ort_session:
                try:
                    entities.extend(self._detect_onnx_image(input_data))
                except Exception as e:
                    print(f"ONNX image detection error: {e}")
            if self.pt_model:
                try:
                    entities.extend(self._detect_pytorch_image(input_data))
                except Exception as e:
                    print(f"PyTorch image detection error: {e}")
        else:
            # Text-based detection (regex fallback)
            entities.extend(self._regex_detection(input_data))

        print(f"Detected entities: {entities}")  # Debug logging
        return self._remove_overlaps(entities)

    def _detect_onnx_image(self, image: Image.Image) -> List[Dict]:
        """Detect PII in image using ONNX model"""
        # Preprocess image for ONNX (fixed to 640x640)
        img_array = np.array(image.resize((640, 640))) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = img_array.astype(np.float32)[None, ...]  # Add batch dimension
        print(f"ONNX input shape: {img_array.shape}")  # Debug logging

        try:
            # Run inference
            results = self.ort_session.run(None, {"images": img_array})
            return self._parse_yolo_results(results, image.size, is_onnx=True)
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return []

    def _detect_pytorch_image(self, image: Image.Image) -> List[Dict]:
        """Detect PII in image using PyTorch YOLO model"""
        try:
            # Enforce 640x640 input size and lower confidence threshold
            results = self.pt_model.predict(image, imgsz=(640, 640), conf=0.01, verbose=True)
            print(f"PyTorch input image size: {results[0].orig_shape}, processed size: {results[0].boxes.xyxy.shape}")  # Debug logging
            return self._parse_yolo_results(results, image.size, is_onnx=False)
        except Exception as e:
            print(f"PyTorch image detection error: {e}")
            return []

    def _parse_yolo_results(self, results, original_size: tuple, is_onnx: bool = False) -> List[Dict]:
        """Parse YOLO model outputs"""
        entities = []
        img_width, img_height = original_size

        if is_onnx:
            # ONNX output: assuming [1, num_boxes, 4 + num_classes]
            output = results[0]  # Shape: [1, num_boxes, 4 + num_classes]
            boxes = output[:, :, :4]  # [x1, y1, x2, y2]
            scores = output[:, :, 4]  # Confidence scores
            labels = np.argmax(output[:, :, 5:], axis=-1)  # Class indices
            boxes = boxes[0]  # Remove batch dimension
            scores = scores[0]
            labels = labels[0]
            scale_w, scale_h = img_width / 640, img_height / 640
        else:
            # PyTorch YOLO results
            boxes = []
            scores = []
            labels = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                # Scale factors for PyTorch
                scale_w, scale_h = img_width / result.orig_shape[1], img_height / result.orig_shape[0]

        print(f"Raw detections - Boxes: {boxes}, Scores: {scores}, Labels: {labels}")  # Debug logging

        for box, score, label in zip(boxes, scores, labels):
            # Scale bounding boxes to original image size
            x1, y1, x2, y2 = box
            x1, x2 = x1 * scale_w, x2 * scale_w
            y1, y2 = y1 * scale_h, y2 * scale_h

            # Validate and correct bounding box
            x1, x2 = min(x1, x2), max(x1, x2)  # Ensure x2 >= x1
            y1, y2 = min(y1, y2), max(y1, y2)  # Ensure y2 >= y1
            x1, y1 = max(0, x1), max(0, y1)  # Clamp to image bounds
            x2, y2 = min(img_width - 1, x2), min(img_height - 1, y2)  # Clamp to image bounds

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                print(f"Skipping invalid box: [{x1}, {y1}, {x2}, {y2}]")
                continue

            # Map label to PII type (adjust based on your model's classes)
            pii_type = {0: "EMAIL", 1: "PHONE", 2: "SSN", 3: "NAME"}.get(int(label), "UNKNOWN")
            entities.append({
                "type": pii_type,
                "value": "",
                "start": int(x1),
                "end": int(x2),
                "score": float(score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        return entities

    def _regex_detection(self, text: str) -> List[Dict]:
        """Fallback regex detection for text"""
        patterns = {
            "EMAIL": r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}',
            "PHONE": r'(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "NAME": r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
        }
        
        entities = []
        for pii_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.9
                })
        return entities

    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities"""
        if not entities:
            return []
        
        entities.sort(key=lambda x: (-x['score'], x['start']))
        filtered = []
        max_end = max(e['end'] for e in entities) if entities else 0
        occupied = [False] * (max_end + 1)
        
        for entity in entities:
            if not any(occupied[entity['start']:entity['end']]):
                filtered.append(entity)
                occupied[entity['start']:entity['end']] = [True] * (entity['end'] - entity['start'])
        
        return filtered