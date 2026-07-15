from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8s.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect_image(self, img, conf=0.25):
        results = self.model.predict(img, conf=conf, verbose=False)
        return results[0]

    def track_frame(self, frame, conf=0.25):
        # Use ByteTrack for robust object tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)
        return results[0]
