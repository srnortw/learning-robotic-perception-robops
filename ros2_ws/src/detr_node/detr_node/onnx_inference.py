import numpy as np
import onnxruntime as ort
import cv2

INPUT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
INPUT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = (800, 800)


class DetrOnnxInference:
    def __init__(self, model_path: str, class_names: list[str] | None = None):
        """
        Args:
            model_path:   Path to model_int8.onnx
            class_names:  Ordered list of class names matching the model's output IDs.
                          Provided by detr_node from detr_params.yaml → class_names,
                          which is auto-synced from label_schema.yaml via schema.py.
                          Falls back to numeric IDs if not provided.
        """
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.class_names = class_names or []

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
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
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
