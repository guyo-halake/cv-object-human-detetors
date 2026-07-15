import gradio as gr
import cv2
import tempfile
import os
import numpy as np

from detection.detector import ObjectDetector
from tracking.tracker import TrajectoryTracker
from counting.counter import Counter
from export.exporter import DataExporter

detector = ObjectDetector("yolov8s.pt")
CLASS_NAMES = detector.class_names
CLASS_CHOICES = list(CLASS_NAMES.values())

def process_video(video_path, selected_classes, line_enabled, lx1, ly1, lx2, ly2):
    if video_path is None:
        return None, None, "No video provided."
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    counter = Counter(CLASS_NAMES)
    if line_enabled:
        p1 = (int(lx1 * width), int(ly1 * height))
        p2 = (int(lx2 * width), int(ly2 * height))
        counter.set_line(p1, p2)
        
    exporter = DataExporter()
    
    selected_class_ids = [k for k, v in CLASS_NAMES.items() if v in selected_classes]
    
    frame_num = 0
    while cap.isOpened():
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
                
        exporter.log_frame(frame_num, frame_num / fps, counter.get_counts())
        
        counts = counter.get_counts()
        y_offset = 30
        for cls in selected_classes:
            text = f"{cls}: {counts['global'].get(cls, 0)}"
            if line_enabled:
                text += f" (In: {counts['in'].get(cls, 0)}, Out: {counts['out'].get(cls, 0)})"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            y_offset += 30
            
        out.write(frame)
        frame_num += 1
        
    cap.release()
    out.release()
    
    csv_path = exporter.export_csv()
    
    final_counts = counter.get_counts()
    count_summary = "### Final Counts\n"
    for cls in selected_classes:
        count_summary += f"- **{cls}**: {final_counts['global'].get(cls, 0)}"
        if line_enabled:
            count_summary += f" (In: {final_counts['in'].get(cls, 0)}, Out: {final_counts['out'].get(cls, 0)})"
        count_summary += "\n"
        
    return out_path, csv_path, count_summary

def process_image(img, selected_classes):
    if img is None:
        return None, "No image provided."
        
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = detector.detect_image(img_bgr)
    
    selected_class_ids = [k for k, v in CLASS_NAMES.items() if v in selected_classes]
    
    counts = {}
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.int().cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()
        
        for box, class_id, conf in zip(boxes, class_ids, confs):
            if class_id not in selected_class_ids:
                continue
                
            cls_name = CLASS_NAMES[class_id]
            counts[cls_name] = counts.get(cls_name, 0) + 1
            
            x1, y1, x2, y2 = map(int, box)
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    count_summary = "### Counts\n"
    for cls, c in counts.items():
        count_summary += f"- **{cls}**: {c}\n"
        
    return img_rgb, count_summary

with gr.Blocks(title="Matta-Vision") as demo:
    gr.Markdown("# Matta-Vision: Object Detection, Tracking & Counting")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload or Record Video", sources=["upload", "webcam"])
            input_image = gr.Image(label="Upload or Snap Image", sources=["upload", "clipboard", "webcam"])
            
            classes = gr.Dropdown(choices=CLASS_CHOICES, value=["person"], multiselect=True, label="Classes to Track")
            
            with gr.Accordion("Line Crossing Setup (Video Only)", open=False):
                line_enabled = gr.Checkbox(label="Enable Line Crossing", value=False)
                lx1 = gr.Slider(0, 1, value=0.1, label="Line Start X (Ratio)")
                ly1 = gr.Slider(0, 1, value=0.5, label="Line Start Y (Ratio)")
                lx2 = gr.Slider(0, 1, value=0.9, label="Line End X (Ratio)")
                ly2 = gr.Slider(0, 1, value=0.5, label="Line End Y (Ratio)")
                
            btn = gr.Button("Process Video", variant="primary")
            btn_img = gr.Button("Process Image")
            
        with gr.Column():
            output_video = gr.Video(label="Output Video")
            output_image = gr.Image(label="Output Image")
            counts_display = gr.Markdown("### Counts will appear here")
            csv_file = gr.File(label="Download CSV Log")
            
    btn.click(
        fn=process_video,
        inputs=[input_video, classes, line_enabled, lx1, ly1, lx2, ly2],
        outputs=[output_video, csv_file, counts_display]
    )
    
    btn_img.click(
        fn=process_image,
        inputs=[input_image, classes],
        outputs=[output_image, counts_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
