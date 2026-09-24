"""Prueba de vida pasiva con un ensamble MiniFASNet V2 + V1SE en ONNX."""
from collections import deque
import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_HASHES = {
    "MiniFASNetV2.onnx": "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
    "MiniFASNetV1SE.onnx": "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
}


class MiniFASModel:
    def __init__(self, path, crop_scale):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Falta el modelo anti-spoofing: {path.name}")
        expected_hash = MODEL_HASHES.get(path.name)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash is None or actual_hash != expected_hash:
            raise RuntimeError(f"El modelo anti-spoofing no superó SHA256: {path.name}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.crop_scale = crop_scale

    @staticmethod
    def _softmax(values):
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def _crop(self, image, face):
        x, y, width, height = [float(value) for value in face[:4]]
        image_height, image_width = image.shape[:2]
        if width <= 1 or height <= 1:
            raise ValueError("Caja facial inválida para prueba de vida")
        scale = min(
            (image_height - 1) / height,
            (image_width - 1) / width,
            self.crop_scale,
        )
        crop_width, crop_height = width * scale, height * scale
        center_x, center_y = x + width / 2, y + height / 2
        x1 = max(0, int(center_x - crop_width / 2))
        y1 = max(0, int(center_y - crop_height / 2))
        x2 = min(image_width - 1, int(center_x + crop_width / 2))
        y2 = min(image_height - 1, int(center_y + crop_height / 2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Recorte facial vacío para prueba de vida")
        return cv2.resize(image[y1:y2 + 1, x1:x2 + 1], (80, 80))

    def probabilities(self, image, face):
        crop = self._crop(image, face).astype(np.float32)
        tensor = np.expand_dims(np.transpose(crop, (2, 0, 1)), axis=0)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self._softmax(logits)[0]


class LivenessDecision:
    """Suaviza resultados y evita aceptar o rechazar por un solo frame."""

    def __init__(self, window=7, minimum_samples=4, real_threshold=0.70, fake_threshold=0.20):
        self.scores = deque(maxlen=window)
        self.minimum_samples = minimum_samples
        self.real_threshold = real_threshold
        self.fake_threshold = fake_threshold

    def reset(self):
        self.scores.clear()

    def update(self, real_score):
        self.scores.append(float(real_score))
        average = float(np.mean(self.scores))
        if len(self.scores) < self.minimum_samples:
            return "verificando", average
        real_votes = sum(score >= self.real_threshold for score in self.scores)
        fake_votes = sum(score <= self.fake_threshold for score in self.scores)
        if real_votes >= 4 and average >= self.real_threshold:
            return "real", average
        if fake_votes >= 3 and average <= self.fake_threshold:
            return "falso", average
        return "inconcluso", average


class AntiSpoofingEngine:
    def __init__(self, models_directory, real_threshold=0.70, fake_threshold=0.20):
        models_directory = Path(models_directory)
        self.models = (
            MiniFASModel(models_directory / "MiniFASNetV2.onnx", 2.7),
            MiniFASModel(models_directory / "MiniFASNetV1SE.onnx", 4.0),
        )
        self.decision = LivenessDecision(
            real_threshold=real_threshold, fake_threshold=fake_threshold
        )

    def reset(self):
        self.decision.reset()

    def analyze(self, image, face):
        probabilities = [model.probabilities(image, face) for model in self.models]
        combined = np.mean(probabilities, axis=0)
        # En MiniFASNet la clase 1 representa una presentación real.
        raw_real_score = float(combined[1])
        state, stable_score = self.decision.update(raw_real_score)
        return state, stable_score, raw_real_score
