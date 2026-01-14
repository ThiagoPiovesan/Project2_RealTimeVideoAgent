import sys
import cv2
import time
import numpy as np
import concurrent.futures

from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path

# Add agent module to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent.video_analytic_agent import VideoAnalyticAgent

video_agent = VideoAnalyticAgent()

# Load the YOLO26 model
model = YOLO("../model/yolo26s.pt")

# Open the video file
video_path = "../data/video.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
track_time = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO26 tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            # Visualize the result on the frame
            frame = result.plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                current_time = time.time()
                # if track_id not in track_time or (current_time - track_time[track_id]) > 5:
                if track_id not in track_time:
                    track_time[track_id] = current_time
                
                if (current_time - track_time[track_id]) > 5:
                    track_time[track_id] = current_time
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future1 = executor.submit(video_agent.analyze_video_frame, frame, box, track_id)

                        result1 = future1.result()
                        
                        print(f"Result 1: {result1}")
                        
                # Update the track history
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLO26 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()