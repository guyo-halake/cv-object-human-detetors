# Matta-Vision

See, label, and count — real-time object and person detection with persistent tracking, so nothing gets counted twice.

![Matta-Vision Demo](sample.gif)


## What it does
- **Object/person detection:** Upload images or video to detect objects with bounding boxes, labels, and confidence scores.
- **Multi-class differentiation:** Distinguishes between different object types (e.g., person, car, animal).
- **Persistent tracking (video):** Each detected object gets a stable ID that persists as it moves.
- **Live counting:** Running totals update in real-time as the video plays.
- **Directional counting:** Counts objects crossing a user-defined virtual line, split by direction (in vs. out).
- **Configurable class filter:** Select which object classes to detect and count.
- **Exportable results:** Annotated output video/image and a CSV log of counts over time.




## Design Decisions
- **Pretrained YOLOv8:** Used a proven, free pretrained model (COCO) to focus engineering effort on tracking and counting logic. Fine-tuning on a custom dataset (e.g., livestock) is a natural next step for domain-specific deployment.
- **ByteTrack:** Used a proven, maintained tracker rather than re-implementing tracking from scratch. It solves the real counting problem effectively and efficiently.
- **Gradio over Streamlit:** Gradio's native video component handles media smoothly and is the more common choice for CV demos on Hugging Face Spaces.



