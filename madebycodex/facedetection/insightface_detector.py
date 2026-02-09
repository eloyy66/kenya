"""Detector de rostro y landmarks usando InsightFace."""

from insightface.app import FaceAnalysis


class InsightFaceDetector:
    """Wrapper ligero de InsightFace."""

    def __init__(self, detect_size=(320, 320), providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "landmark_3d_68", "landmark_2d_106"],
        )
        self.app.prepare(ctx_id=-1, det_size=detect_size)

    def detect(self, frame):
        """Retorna lista de rostros detectados."""
        return self.app.get(frame)
