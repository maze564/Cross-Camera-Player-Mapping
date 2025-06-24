import cv2
import torch
import os
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

def extract_color_histogram(image, bbox):
    x1, y1, x2, y2 = bbox
    cropped = image[int(y1):int(y2), int(x1):int(x2)]
    hist = cv2.calcHist([cropped], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()

def run_detection(video_name):
    model = YOLO("yolov11/best.pt")
    tracker = DeepSort()

    cap = cv2.VideoCapture(f"videos/{video_name}")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("outputs", exist_ok=True)
    out = cv2.VideoWriter(f"outputs/{video_name}_tracked.avi",
                          cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    features = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        results = model(frame)[0]
        for result in results.boxes:
            if int(result.cls[0]) == 0:  # 0 = person
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                conf = float(result.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = str(track.track_id)
            l, t, r, b = track.to_ltrb()
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(l), int(t) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if track_id not in features:
                features[track_id] = extract_color_histogram(frame, (l, t, r, b))

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    with open(f"outputs/{video_name}_features.json", "w") as f:
        json.dump(features, f)

    print(f"Tracking done for {video_name}, video and features saved.")

if __name__ == "__main__":
    run_detection("broadcast.mp4")
    run_detection("tacticam.mp4")
              
