import cv2
import argparse
from detection.detector import ObjectDetector
from counting.counter import Counter

def main():
    parser = argparse.ArgumentParser(description="Live webcam object tracking and counting.")
    parser.add_argument("--classes", type=str, nargs="+", default=["person"], help="Classes to track (e.g. person car cow).")
    parser.add_argument("--line", action="store_true", help="Enable line crossing counting.")
    args = parser.parse_args()
    
    print("Initializing model...")
    # Use yolov8n.pt for faster real-time performance on webcam
    detector = ObjectDetector("yolov8n.pt") 
    CLASS_NAMES = detector.class_names
    
    selected_class_ids = [k for k, v in CLASS_NAMES.items() if v in args.classes]
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    counter = Counter(CLASS_NAMES)
    
    line_enabled = args.line
    if line_enabled:
        p1 = (int(0.1 * width), int(0.5 * height))
        p2 = (int(0.9 * width), int(0.5 * height))
        counter.set_line(p1, p2)
        
    print(f"Starting webcam. Tracking classes: {args.classes}. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = detector.track_frame(frame)
        
        if line_enabled:
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
            
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            class_ids = results.boxes.cls.int().cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()
            
            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                if class_id not in selected_class_ids:
                    continue
                    
                counter.update(track_id, class_id, box)
                
                x1, y1, x2, y2 = map(int, box)
                cls_name = CLASS_NAMES[class_id]
                label = f"{cls_name} {track_id} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # Draw counts
        counts = counter.get_counts()
        y_offset = 30
        for cls in args.classes:
            text = f"{cls}: {counts['global'].get(cls, 0)}"
            if line_enabled:
                text += f" (In: {counts['in'].get(cls, 0)}, Out: {counts['out'].get(cls, 0)})"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30
            
        cv2.imshow("Matta-Vision Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
