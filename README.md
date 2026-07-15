# Matta-Vision

See, label, and count — real-time object and person detection with persistent tracking, so nothing gets counted twice.

![Matta-Vision Demo](sample.gif)

## Problem / Use Case
Whether it's footfall counting in a retail space, traffic counting on a highway, or livestock counting from a fixed camera in Marsabit, accurately counting objects as they move across a frame requires more than just drawing bounding boxes. Raw detection counts the same object multiple times as it moves. Matta-Vision solves this by layering persistent object tracking (assigning a stable ID to each object) on top of state-of-the-art YOLOv8 detection. This ensures accurate running counts without double-counting.

## What it does
- **Object/person detection:** Upload images or video to detect objects with bounding boxes, labels, and confidence scores.
- **Multi-class differentiation:** Distinguishes between different object types (e.g., person, car, animal).
- **Persistent tracking (video):** Each detected object gets a stable ID that persists as it moves.
- **Live counting:** Running totals update in real-time as the video plays.
- **Directional counting:** Counts objects crossing a user-defined virtual line, split by direction (in vs. out).
- **Configurable class filter:** Select which object classes to detect and count.
- **Exportable results:** Annotated output video/image and a CSV log of counts over time.

## Architecture
```text
[Video/Image input] 
       ↓
[YOLOv8 detection per frame] 
       ↓
[ByteTrack: persistent IDs]
       ↓
[Class filter] 
       ↓
[Counting logic (+ optional line-crossing direction)]
       ↓
[Annotated output + CSV log]
```

## Tech Stack
| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python | Ultralytics/YOLO ecosystem is Python-native, best tooling |
| Detection model | YOLOv8 (Ultralytics) | Pretrained on COCO (80 classes), free, state-of-the-art for the cost |
| Tracking | ByteTrack | Built into Ultralytics, near-zero extra code for persistent IDs |
| Video/image handling | OpenCV | Standard for frame-by-frame video processing and overlays |
| UI | Gradio | Handles video upload and playback natively, fast to wire up |
| Deployment | Hugging Face Spaces | Handles video/model demos and heavier processing better than static hosting |

## Setup
```bash
pip install -r requirements.txt
python app.py
```

## Design Decisions
- **Pretrained YOLOv8:** Used a proven, free pretrained model (COCO) to focus engineering effort on tracking and counting logic. Fine-tuning on a custom dataset (e.g., livestock) is a natural next step for domain-specific deployment.
- **ByteTrack:** Used a proven, maintained tracker rather than re-implementing tracking from scratch. It solves the real counting problem effectively and efficiently.
- **Gradio over Streamlit:** Gradio's native video component handles media smoothly and is the more common choice for CV demos on Hugging Face Spaces.

## Roadmap
- Custom fine-tuning on domain-specific data (e.g., livestock).
- Live webcam mode for real-time monitoring.
- Multi-camera aggregation for wider coverage areas.
- Time-windowed analytics dashboards.

## License
MIT
