import numpy as np
import onnxruntime as ort
import cv2

# COCO 80-class labels (indices 0-79 match DETR output)
COCO_LABELS = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush',
]

INPUT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
INPUT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = (800, 800)


class DetrOnnxInference:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, INPUT_SIZE)
        normalized = (resized.astype(np.float32) / 255.0 - INPUT_MEAN) / INPUT_STD
        # HWC → CHW → NCHW
        return normalized.transpose(2, 0, 1)[np.newaxis, :]

    def run(self, bgr_frame: np.ndarray) -> list[dict]:
        """Returns list of detections: {label, confidence, box: [cx,cy,w,h] normalized}"""
        tensor = self.preprocess(bgr_frame)
        outputs = self.session.run(None, {self.input_name: tensor})

        # Conditional DETR outputs: logits [1, num_queries, num_classes+1], boxes [1, num_queries, 4]
        logits, boxes = outputs[0][0], outputs[1][0]

        scores = self._softmax(logits)
        # Last class is 'no object' — exclude it
        class_scores = scores[:, :-1]
        confidences = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        detections = []
        for i, (conf, cls_id, box) in enumerate(zip(confidences, class_ids, boxes)):
            if conf < 0.1:
                continue
            label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else str(cls_id)
            detections.append({
                'label': label,
                'class_id': int(cls_id),
                'confidence': float(conf),
                'box': box.tolist(),  # [cx, cy, w, h] normalized 0-1
            })

        return detections

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
