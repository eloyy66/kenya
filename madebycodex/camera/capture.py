"""Captura de video con OpenCV."""

import cv2


class Camera:
    """Encapsula la captura de video."""

    def __init__(self, index=0, backend="DEFAULT", width=640, height=360):
        self.index = index
        self.backend = backend
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """Abre la camara con el backend configurado."""
        if self.backend == "AVFOUNDATION":
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def warmup(self, frames=10):
        """Descarta frames iniciales para estabilizar la camara."""
        if self.cap is None:
            return
        for _ in range(frames):
            self.cap.read()

    def read(self):
        """Lee un frame de la camara."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        """Libera la camara."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
