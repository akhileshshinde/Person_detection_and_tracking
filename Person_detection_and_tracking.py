!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
!pip install -U yolov5
!pip install opencv-python-headless numpy dlib norfair
!pip install yt-dlp

import cv2
import torch
import numpy as np
from norfair import Detection, Tracker
from google.colab.patches import cv2_imshow
from google.colab import files
import subprocess

#Function to download YouTube video
def download_youtube_video(video_url, output_filename="video.mp4"):
    subprocess.run(["yt-dlp", "-f", "best[ext=mp4]", "-o", output_filename, video_url])
    print(f"Video downloaded: {output_filename}")

video_url = "https://youtu.be/ORrrKXGx2SE"
video_path = '/content/video.mp4'
download_youtube_video(video_url, video_path)

# for uploading from your device
# uploaded = files.upload()
# video_path = '/content/video.mp4'

cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    print("Video loaded successfully.")
else:
    print("Error: Could not open video.")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

tracker = Tracker(distance_function="euclidean", distance_threshold=50)

output_path = '/content/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform detection

    detections = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()

        if int(cls) == 0 and conf > 0.5:
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            detections.append(Detection(points=np.array([[x_center, y_center]]), scores=np.array([conf])))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        x, y = obj.estimate[0]
        cv2.putText(frame, f'ID: {obj.id}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out.write(frame)

    # this displays the frame with bounding boxes and IDs , if you want to see frames (optional)
    cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at {output_path}")
