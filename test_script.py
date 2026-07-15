import cv2
import numpy as np
from app import process_image, process_video

print("Testing Image...")
img = np.zeros((480, 640, 3), dtype=np.uint8)
img_rgb, summary = process_image(img, ["person"])
print("Image output shape:", img_rgb.shape)

print("Testing Video...")
# create a dummy video
out = cv2.VideoWriter('dummy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
for i in range(10):
    out.write(np.zeros((480, 640, 3), dtype=np.uint8))
out.release()
video_out, csv_out, summary_vid = process_video('dummy.mp4', ["person"], False, 0,0,0,0)
print("Video processed:", video_out, csv_out)
