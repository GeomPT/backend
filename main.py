from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time  # Import time module for countdown
from datetime import datetime
from opencv_logic import (
    process_frame as measure_process_frame,
    USE_CONFIDENCE_THRESHOLD,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

client_processing_options = {}
client_pose_instances = {}
client_measurement_state = {}

# Booleans to toggle levels of rigor on the auto-measurement
USE_COUNTDOWN = True  
USE_ANGLE_SMOOTHING = True  
USE_MEASUREMENT_DELAY = True # False ends measurement immediately when angle drops
# USE_CONFIDENCE_THRESHOLD is imported from opencv_logic.py


@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}")
    # Initialize Pose instance for the client
    pose_instance = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    client_pose_instances[request.sid] = pose_instance


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

    # Measurement logic
    measurement_state = client_measurement_state.get(request.sid)
    if measurement_state:
        if USE_COUNTDOWN and measurement_state["countdown_started"]:
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
                    f"{int(remaining_time) + 1}",  # +1 to account for floor rounding
                    (
                        int(processed_frame.shape[1] / 2),
                        int(processed_frame.shape[0] / 2),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (0, 0, 0),
                    6,
                )
        elif not USE_COUNTDOWN or measurement_state["measurement_started"]:
            if angle is not None:
                # Update angle history
                if USE_ANGLE_SMOOTHING:
                    angle_history = measurement_state["angle_history"]
                    angle_history.append(angle)
                    if len(angle_history) > measurement_state["angle_history_size"]:
                        angle_history.pop(0)  # Remove oldest angle
                    # Calculate average angle
                    smoothed_angle = sum(angle_history) / len(angle_history)
                else:
                    smoothed_angle = angle

                current_max_angle = measurement_state["max_angle"]
                if current_max_angle is None or smoothed_angle > current_max_angle:
                    # Update max_angle and store the frame
                    measurement_state["max_angle"] = smoothed_angle
                    measurement_state["max_angle_frame"] = processed_frame.copy()
                    measurement_state["below_threshold_counter"] = 0  # Reset counter
                    # Store confidence
                    measurement_state["max_angle_confidence"] = confidence
                elif current_max_angle - smoothed_angle >= 20:
                    if USE_MEASUREMENT_DELAY:
                        # Increment the counter
                        measurement_state["below_threshold_counter"] += 1
                        if (
                            measurement_state["below_threshold_counter"]
                            >= measurement_state["required_below_threshold_frames"]
                        ):
                            # Angle has been below threshold for required number of frames
                            measurement_state["measurement_started"] = False
                            # Save the frame and measurement result
                            save_measurement(measurement_state, request.sid)
                            # Remove measurement state
                            client_measurement_state.pop(request.sid, None)
                            # Notify client
                            socketio.emit(
                                "measurement_completed",
                                {"message": "Measurement completed"},
                                to=request.sid,
                            )
                    else:
                        # Directly end measurement
                        measurement_state["measurement_started"] = False
                        # Save the frame and measurement result
                        save_measurement(measurement_state, request.sid)
                        # Remove measurement state
                        client_measurement_state.pop(request.sid, None)
                        # Notify client
                        socketio.emit(
                            "measurement_completed",
                            {"message": "Measurement completed"},
                            to=request.sid,
                        )
                else:
                    # Angle is above threshold again, reset the counter
                    measurement_state["below_threshold_counter"] = 0
            else:
                # Handle missing angles (e.g., landmarks not detected)
                measurement_state["missing_landmarks_counter"] += 1
                if (
                    measurement_state["missing_landmarks_counter"]
                    >= measurement_state["max_missing_landmarks_frames"]
                ):
                    # Consider resetting or pausing the measurement
                    measurement_state["measurement_started"] = False
                    print(
                        f"Measurement paused due to missing landmarks for client {request.sid}"
                    )
                    # Optionally, notify the client
                    socketio.emit(
                        "measurement_paused",
                        {"message": "Measurement paused due to missing landmarks"},
                        to=request.sid,
                    )
        else:
            # Measurement has ended or paused
            pass

    # Encode frame as JPEG with maximum quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode(".jpg", processed_frame, encode_param)
    frame_data = buffer.tobytes()

    socketio.emit("processed_frame", frame_data, to=request.sid)


@socketio.on("begin_measurement")
def handle_begin_measurement():
    # Initialize measurement state for the client
    client_measurement_state[request.sid] = {
        "measurement_started": (
            False if USE_COUNTDOWN else True
        ),  # Measurement starts after countdown if enabled
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
        )
        angle = None
        confidence = None
    return frame, angle, confidence


def save_measurement(measurement_state, client_id):
    max_angle = measurement_state["max_angle"]
    max_angle_frame = measurement_state["max_angle_frame"]
    joint_name = measurement_state["joint_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{joint_name}_{timestamp}_{client_id}"
    max_angle_confidence = measurement_state.get("max_angle_confidence", 0)

    # Check if confidence is high enough to save measurement
    if USE_CONFIDENCE_THRESHOLD and (
        max_angle_confidence is None or max_angle_confidence < 0.5
    ):
        print(f"Measurement not saved due to low confidence for client {client_id}")
        # Optionally notify client
        socketio.emit(
            "measurement_failed",
            {"message": "Measurement not saved due to low confidence"},
            to=client_id,
        )
        return

    photo_folder_path = os.path.join("measurements", filename_base)
    os.makedirs(photo_folder_path, exist_ok=True)

    # Save the image
    image_filename = os.path.join(photo_folder_path, "photo.jpg")
    cv2.imwrite(image_filename, max_angle_frame)

    # Save the measurement result as JSON
    measurement_data = {
        "joint_name": joint_name,
        "max_angle": max_angle,
        "timestamp": timestamp,
        "confidence": max_angle_confidence,
    }
    json_filename = os.path.join(photo_folder_path, "angle.json")

    with open(json_filename, "w") as f:
        json.dump(measurement_data, f)

    print(f"Measurement saved for client {client_id}: angle {max_angle} at {timestamp}")
    socketio.emit(
        "measurement_saved",
        {"message": "Measurement saved successfully"},
        to=client_id,
    )


# 'assert status_set is not None, "write() before start_response"' error is due
# to running locally. Would not appear on production.
# https://github.com/miguelgrinberg/flask-sock/issues/27
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)
