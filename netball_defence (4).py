
from google.colab import drive
drive.mount('/content/drive')

!pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import cv2
import mediapipe as mp
import numpy as np

# Setup mediapipe instance
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_joint_coordinates(landmarks, joint_name, width, height):
    """Convert normalized coordinates to pixel coordinates"""
    landmark = landmarks[joint_name.value]
    return {
        'x': int(landmark.x * width),
        'y': int(landmark.y * height),
        'z': round(landmark.z, 3),
        'visibility': round(landmark.visibility, 2)
    }

def add_title_bar(image, title, bar_height=60, bg_color=(245, 117, 16)):
    """Add a title bar to the top of the image"""
    h, w = image.shape[:2]
    # Create title bar
    title_bar = np.full((bar_height, w, 3), bg_color, dtype=np.uint8)

    # Add text to title bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)

    # Get text size to center it
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2

    cv2.putText(title_bar, title, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Combine title bar with image
    return np.vstack((title_bar, image))

# Initialize VideoCapture
input_path = r"/content/drive/MyDrive/new testing 66/VID-20240907-WA0050 (3).mp4"
output_path = r"/content/drive/MyDrive/new testing 66/output_pose_detection.mp4"
video_title = "Netball Player Joint Analysis - Defence analysis"

cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate dimensions for the combined frame (original video + info display)
combined_width = width + 400  # 400 is the width of info display
title_bar_height = 60  # Height of the title bar
combined_height = height + title_bar_height  # Add title bar height

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

# Dictionary of joints to track
joints_to_track = {


    'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
    'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
    'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
    'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
    'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
    'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE


}

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        if current_frame % 10 == 0:  # Print progress every 10 frames
            print(f"Processing frame {current_frame} of {frame_count} ({(current_frame/frame_count*100):.1f}%)")


        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Create background for joint coordinates display
            info_display = np.zeros((height, 400, 3), dtype=np.uint8)
            y_offset = 30

            # Display joint coordinates
            for joint_name, landmark_id in joints_to_track.items():
                coords = get_joint_coordinates(landmarks, landmark_id, width, height)

                # Display joint information on the side panel
                text = f"{joint_name}: x={coords['x']}, y={coords['y']}, z={coords['z']}"
                cv2.putText(info_display, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 30

                # Mark joint position on the main image
                cv2.circle(image, (coords['x'], coords['y']), 5, (0, 255, 0), -1)
                cv2.putText(image, joint_name, (coords['x'] + 10, coords['y']),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error in frame {current_frame}: {e}")
            pass

        # Render pose detection
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Combine main image and info display
        combined_image = np.hstack((image, info_display))

        # Add title bar to the combined image
        final_image = add_title_bar(combined_image, video_title)

        # Write the frame to output video
        out.write(final_image)

    cap.release()
    out.release()
    print("Video processing completed! Output saved to:", output_path)