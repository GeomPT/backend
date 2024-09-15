from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import os
import time
from datetime import datetime
from collections import deque
import threading
from io import BytesIO
import uuid

from opencv_logic import (
    process_frame as measure_process_frame,
    USE_CONFIDENCE_THRESHOLD,
)
from firebase_util import loadFirebaseFromApp, save_file_to_storage, save_measurement_to_firestore
from firebase_admin import firestore

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
CORS(app, origins=["http://localhost:3000"])  # Flask CORS needed for DB
# socketio cors needed for websocket
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

client_processing_options = {}
client_pose_instances = {}
client_measurement_state = {}
client_frame_buffers = {}
client_user_info = {}  # Stores userName and workout for each client

USE_COUNTDOWN = True
USE_ANGLE_SMOOTHING = True
USE_MEASUREMENT_DELAY = True

# Frame buffering and video saving configurations
PRE_MEASUREMENT_SECONDS = 2  # Seconds before measurement
POST_MEASUREMENT_SECONDS = 1  # Seconds after measurement
FRAME_RATE = 30

PRE_FRAME_BUFFER_SIZE = int(PRE_MEASUREMENT_SECONDS * FRAME_RATE)
POST_FRAME_BUFFER_SIZE = int(POST_MEASUREMENT_SECONDS * FRAME_RATE)

# Video saving configuration
VIDEO_FOLDER = "videos"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Initialize Firebase
loadFirebaseFromApp(app)
db = firestore.client()

@app.route("/")
def index():
    return "Welcome to the Measurement Server!"


@app.route("/<path:path>")
def serve_static_file(path):
    return send_from_directory("static", path)


@socketio.on("connect")
def handle_connect(auth):
    print(f"Client connected: {request.sid}")

    # Extract userName and workout from auth object
    user_name = "Bob_8f0c3aae-30ce-4c6d-b6d1-0c3993e1808d"
    workout = "knee1" #send a workout type

    if not user_name or not workout:
        print(f"Missing userName or workout for client {request.sid}")
        emit('connection_error', {'message': 'Missing userName or workout'})
        return

    # Store user info
    client_user_info[request.sid] = {
        'userName': user_name,
        'workout': workout
    }

    # Initialize Pose instance for the client
    pose_instance = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    client_pose_instances[request.sid] = pose_instance
    # Initialize pre-measurement frame buffer for the client
    client_frame_buffers[request.sid] = deque(maxlen=PRE_FRAME_BUFFER_SIZE)



@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    client_processing_options.pop(request.sid, None)
    # Close and remove the Pose instance for the client
    pose_instance = client_pose_instances.pop(request.sid, None)
    if pose_instance:
        pose_instance.close()
    # Remove measurement state
    client_measurement_state.pop(request.sid, None)
    # Remove frame buffer
    client_frame_buffers.pop(request.sid, None)
    # Remove user info
    client_user_info.pop(request.sid, None)


@socketio.on("start_processing")
def handle_start_processing(data):
    processing_type = data.get("processingType", "default")
    client_processing_options[request.sid] = processing_type
    print(f"Client {request.sid} requested processing type: {processing_type}")


@socketio.on("send_frame")
def handle_send_frame(frame_data):
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Failed to decode frame from client {request.sid}")
        return

    processing_type = client_processing_options.get(request.sid, "default")
    pose_instance = client_pose_instances.get(request.sid)
    if pose_instance is None:
        print(f"No Pose instance for client {request.sid}")
        return

    processed_frame, angle, confidence = process_frame(
        frame, processing_type, pose_instance
    )

    # Add the processed frame to the pre-measurement buffer
    if request.sid in client_frame_buffers:
        client_frame_buffers[request.sid].append(processed_frame.copy())
    else:
        client_frame_buffers[request.sid] = deque(maxlen=PRE_FRAME_BUFFER_SIZE)
        client_frame_buffers[request.sid].append(processed_frame.copy())

    # Measurement logic
    measurement_state = client_measurement_state.get(request.sid)
    if measurement_state:
        if USE_COUNTDOWN and measurement_state.get("countdown_started", False):
            if measurement_state["countdown_start_time"] is None:
                measurement_state["countdown_start_time"] = time.time()
            elapsed_time = time.time() - measurement_state["countdown_start_time"]
            remaining_time = measurement_state["countdown_time"] - elapsed_time
            if remaining_time <= 0:
                measurement_state["countdown_started"] = False
                measurement_state["measurement_started"] = True
                print(f"Measurement started for client {request.sid}")
            else:
                # Draw countdown on the frame
                cv2.putText(
                    processed_frame,
                    f"{int(remaining_time) + 1}",
                    (
                        int(processed_frame.shape[1] / 2),
                        int(processed_frame.shape[0] / 2),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (0, 0, 255),
                    6,
                    cv2.LINE_AA,
                )
        elif not USE_COUNTDOWN or measurement_state.get("measurement_started", False):
            if angle is not None:
                if USE_ANGLE_SMOOTHING:
                    angle_history = measurement_state["angle_history"]
                    angle_history.append(angle)
                    if len(angle_history) > measurement_state["angle_history_size"]:
                        angle_history.pop(0)
                    smoothed_angle = sum(angle_history) / len(angle_history)
                else:
                    smoothed_angle = angle

                current_max_angle = measurement_state["max_angle"]
                if current_max_angle is None or smoothed_angle > current_max_angle:
                    measurement_state["max_angle"] = smoothed_angle
                    measurement_state["max_angle_frame"] = processed_frame.copy()
                    measurement_state["below_threshold_counter"] = 0
                    measurement_state["max_angle_confidence"] = confidence
                elif current_max_angle - smoothed_angle >= 20:
                    if USE_MEASUREMENT_DELAY:
                        measurement_state["below_threshold_counter"] += 1
                        if (
                            measurement_state["below_threshold_counter"]
                            >= measurement_state["required_below_threshold_frames"]
                        ):
                            measurement_state["measurement_started"] = False
                            save_measurement(measurement_state, request.sid)
                            # Start post-measurement frame collection
                            initiate_post_measurement(request.sid)
                    else:
                        measurement_state["measurement_started"] = False
                        save_measurement(measurement_state, request.sid)
                        # Start post-measurement frame collection
                        initiate_post_measurement(request.sid)
                else:
                    measurement_state["below_threshold_counter"] = 0
            else:
                measurement_state["missing_landmarks_counter"] += 1
                if (
                    measurement_state["missing_landmarks_counter"]
                    >= measurement_state["max_missing_landmarks_frames"]
                ):
                    measurement_state["measurement_started"] = False
                    print(
                        f"Measurement failed due to missing landmarks for client {request.sid}"
                    )
                    client_measurement_state.pop(request.sid, None)
                    socketio.emit(
                        "measurement_failed",
                        {"message": "Measurement failed due to missing landmarks"},
                        to=request.sid,
                    )

        # Handle post-measurement frame collection
        if measurement_state.get("post_measurement_started", False):
            elapsed_time = (
                time.time() - measurement_state["post_measurement_start_time"]
            )
            if elapsed_time <= POST_MEASUREMENT_SECONDS:
                # Collect post-measurement frames
                measurement_state["post_measurement_frames"].append(
                    processed_frame.copy()
                )
            else:
                # Post-measurement frame collection completed
                frames_to_save = (
                    measurement_state["pre_measurement_frames"]
                    + measurement_state["post_measurement_frames"]
                )
                threading.Thread(
                    target=save_mp4_video, args=(frames_to_save, request.sid, measurement_state)
                ).start()
                # Clean up measurement state
                client_measurement_state.pop(request.sid, None)

    # Encode frame as JPEG with maximum quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode(".jpg", processed_frame, encode_param)
    frame_data_encoded = buffer.tobytes()

    socketio.emit("processed_frame", frame_data_encoded, to=request.sid)


@socketio.on("begin_measurement")
def handle_begin_measurement():
    # Initialize measurement state for the client
    client_measurement_state[request.sid] = {
        "measurement_started": (False if USE_COUNTDOWN else True),
        "max_angle": None,
        "max_angle_frame": None,
        "max_angle_confidence": None,
        "processing_type": client_processing_options.get(request.sid, "default"),
        "joint_name": client_processing_options.get(request.sid, "default"),
        "below_threshold_counter": 0,
        "required_below_threshold_frames": 10 if USE_MEASUREMENT_DELAY else 1,
        "countdown_started": USE_COUNTDOWN,
        "countdown_time": 3,  # Seconds
        "countdown_start_time": None,
        "angle_history": [] if USE_ANGLE_SMOOTHING else None,
        "angle_history_size": 5,  # Number of frames to average over
        "missing_landmarks_counter": 0,
        "max_missing_landmarks_frames": 30,  # Adjust as needed
    }
    print(
        f"Measurement {'countdown started' if USE_COUNTDOWN else 'started'} for client {request.sid}"
    )


def process_frame(frame, processing_type, pose_instance):
    if processing_type in ["knee", "elbow", "shoulder"]:
        frame, angle, confidence = measure_process_frame(
            frame, processing_type, pose_instance
        )
    else:
        cv2.putText(
            frame,
            "Default stream...",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        angle = None
        confidence = None
    return frame, angle, confidence


def save_measurement(measurement_state, client_id):
    max_angle = measurement_state["max_angle"]
    max_angle_frame = measurement_state["max_angle_frame"]
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    measurement_state["timestamp"] = timestamp
    max_angle_confidence = measurement_state.get("max_angle_confidence", 0)

    if USE_CONFIDENCE_THRESHOLD and (
        max_angle_confidence is None or max_angle_confidence < 0.5
    ):
        print(f"Measurement failed due to low confidence for client {client_id}")
        socketio.emit(
            "measurement_failed",
            {"message": "Measurement failed due to low confidence"},
            to=client_id,
        )
        return

    try:
        # Convert image to bytes
        _, img_encoded = cv2.imencode('.jpg', max_angle_frame)
        image_bytes = BytesIO(img_encoded.tobytes())

        image_filename = f"photo_{timestamp}_{client_id}.jpg"

        # Use save_file_to_storage to upload image to Firebase
        image_url = save_file_to_storage(client_id, "image", image_filename, image_bytes)

        # Store image URL and value in measurement state
        measurement_state["image_url"] = image_url
        measurement_state["value"] = max_angle

        print(f"Measurement saved for client {client_id}: angle {max_angle} at {timestamp}")

    except Exception as e:
        print(f"Failed to save measurement for client {client_id}: {e}")
        socketio.emit(
            "measurement_failed",
            {"message": "Failed to save measurement"},
            to=client_id,
        )


def save_mp4_video(frames, client_id, measurement_state):
    if not frames:
        print(f"No frames to save for video for client {client_id}")
        socketio.emit(
            "video_save_failed",
            {"message": "No frames available to save video"},
            to=client_id,
        )
        return

    timestamp = measurement_state.get("timestamp", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    video_filename = f"measurement_{timestamp}_{client_id}.mp4"
    video_path = os.path.join(VIDEO_FOLDER, video_filename)
    try:
        # Increase resolution if needed
        upscale_factor = 1  # Adjust as needed
        frames_resized = [
            cv2.resize(
                frame,
                None,
                fx=upscale_factor,
                fy=upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )
            for frame in frames
        ]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width, _ = frames_resized[0].shape
        video_writer = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (width, height))

        for frame in frames_resized:
            video_writer.write(frame)

        video_writer.release()

        print(f"MP4 video saved for client {client_id} at {video_path}")

        # Read the video as bytes to upload to Firebase
        with open(video_path, "rb") as video_file:
            video_bytes = BytesIO(video_file.read())

        # Save the MP4 video in Firebase Storage
        video_url = save_file_to_storage(client_id, "video", video_filename, video_bytes)

        # Store video URL in measurement state
        measurement_state["video_url"] = video_url

        # Save measurement data to Firestore
        user_info = client_user_info.get(client_id)
        if user_info:
            user_name = user_info['userName']
            workout = user_info['workout']
            measurement_id = str(uuid.uuid4())

            measurement_data = {
                "imageUrl": measurement_state.get("image_url"),
                "timestamp": measurement_state.get("timestamp"),
                "value": measurement_state.get("value"),
                "videoUrl": measurement_state.get("video_url"),
            }

            save_measurement_to_firestore(user_name, workout, measurement_id, measurement_data)

            # Emit event with measurement data
            socketio.emit(
                "measurement_saved",
                {"message": "Measurement completed successfully", "measurement_data": measurement_data},
                to=client_id,
            )

            print(f"Measurement data saved and sent to client {client_id}")

        else:
            print(f"No user info found for client {client_id}")
            socketio.emit(
                "measurement_failed",
                {"message": "User information not found"},
                to=client_id,
            )

    except Exception as e:
        print(f"Failed to save video for client {client_id}: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)
        socketio.emit(
            "video_save_failed",
            {"message": "Failed to save video"},
            to=client_id,
        )


def initiate_post_measurement(client_id):
    measurement_state = client_measurement_state.get(client_id)
    if measurement_state is None:
        print(
            f"No measurement state found for client {client_id} during post-measurement initiation"
        )
        return

    measurement_state["post_measurement_started"] = True
    measurement_state["post_measurement_start_time"] = time.time()
    measurement_state["post_measurement_frames"] = []
    measurement_state["pre_measurement_frames"] = list(
        client_frame_buffers.get(client_id, [])
    )
    # Store timestamp if not already stored
    if "timestamp" not in measurement_state:
        measurement_state["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Initiated post-measurement frame collection for client {client_id}")


if __name__ == "__main__":
    socketio.run(
        app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True
    )
