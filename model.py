import re
import onnxruntime as ort
import torch
from typing import List, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from ultralytics import YOLO

class PIIDetector:
    def __init__(self, onnx_model_path: str = None, pt_model_path: str = None):
        self.ort_session = None
        self.pt_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pii_type = {0: "EMAIL", 1: "PHONE", 2: "SSN", 3: "NAME"}

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

        # Regex patterns for text-based detection
        self.pii_patterns = {
            "EMAIL": r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}',
            "PHONE": r'(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "NAME": r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
        }

    def detect(self, input_data, is_image: bool = False, is_frame: bool = False) -> List[Dict]:
        """Detect PII entities in text, image, or video frame"""
        entities = []

        if is_image or is_frame:
            # Image or frame-based detection
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
            # Text-based detection
            entities.extend(self._regex_detection(input_data))

        print(f"Detected entities: {entities}")
        return self._remove_overlaps(entities)

    def _detect_onnx_image(self, input_data) -> List[Dict]:
        """Detect PII in image or frame using ONNX model"""
        if isinstance(input_data, Image.Image):
            img_array = np.array(input_data.resize((640, 640))) / 255.0
        else:  # Assume OpenCV frame (numpy array)
            img_array = cv2.resize(input_data, (640, 640))
            img_array = img_array.astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = img_array[None, ...]  # Add batch dimension
        print(f"ONNX input shape: {img_array.shape}")

        try:
            results = self.ort_session.run(None, {"images": img_array})
            return self._parse_yolo_results(results, input_data.size if isinstance(input_data, Image.Image) else input_data.shape[:2][::-1], is_onnx=True)
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return []

    def _detect_pytorch_image(self, input_data) -> List[Dict]:
        """Detect PII in image or frame using PyTorch YOLO model"""
        try:
            input_img = input_data if isinstance(input_data, Image.Image) else Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
            results = self.pt_model.predict(input_img, imgsz=(640, 640), conf=0.01, verbose=True)
            print(f"PyTorch input size: {results[0].orig_shape}, processed size: {results[0].boxes.xyxy.shape}")
            return self._parse_yolo_results(results, input_img.size, is_onnx=False)
        except Exception as e:
            print(f"PyTorch image detection error: {e}")
            return []

    def _parse_yolo_results(self, results, original_size: tuple, is_onnx: bool = False) -> List[Dict]:
        """Parse YOLO model outputs"""
        entities = []
        img_width, img_height = original_size

        if is_onnx:
            output = results[0]  # [1, num_boxes, 4 + num_classes]
            boxes = output[:, :, :4]
            scores = output[:, :, 4]
            labels = np.argmax(output[:, :, 5:], axis=-1)
            boxes = boxes[0]
            scores = scores[0]
            labels = labels[0]
            scale_w, scale_h = img_width / 640, img_height / 640
        else:
            boxes = []
            scores = []
            labels = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                scale_w, scale_h = img_width / result.orig_shape[1], img_height / result.orig_shape[0]

        print(f"Raw detections - Boxes: {boxes}, Scores: {scores}, Labels: {labels}")

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            x1, x2 = x1 * scale_w, x2 * scale_w
            y1, y2 = y1 * scale_h, y2 * scale_h
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width - 1, x2), min(img_height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                print(f"Skipping invalid box: [{x1}, {y1}, {x2}, {y2}]")
                continue
            pii_type = self.pii_type.get(int(label), "UNKNOWN")
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
        entities = []
        for pii_type, pattern in self.pii_patterns.items():
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

    def mask_frame(self, frame: np.ndarray, entities: List[Dict], mask_style: str = "black_box") -> np.ndarray:
        """Mask PII in a video frame using PIL with specified style"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for entity in entities:
            bbox = entity.get('bbox', [])
            if not bbox or len(bbox) != 4:
                print(f"Skipping invalid bbox (malformed): {bbox}")
                continue
            x1, y1, x2, y2 = bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.size[0] - 1, x2), min(img.size[1] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                print(f"Skipping invalid bbox after validation: [{x1}, {y1}, {x2}, {y2}]")
                continue
            if mask_style == "black_box":
                draw.rectangle((x1, y1, x2, y2), fill="black")
            elif mask_style == "blur":
                region = img.crop((x1, y1, x2, y2))
                region = region.filter(ImageFilter.GaussianBlur(radius=10))
                img.paste(region, (x1, y1))
            elif mask_style == "pixelation":
                region = img.crop((x1, y1, x2, y2))
                region = region.resize((10, 10), Image.Resampling.NEAREST).resize(region.size, Image.Resampling.NEAREST)
                img.paste(region, (x1, y1))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    detector = PIIDetector(pt_model_path="best.pt")
    # Test with sample image
    img = cv2.imread("test.jpg")
    entities = detector.detect(img, is_image=True)
    masked_frame = detector.mask_frame(img, entities, mask_style="black_box")
    cv2.imwrite("masked_test.jpg", masked_frame)
