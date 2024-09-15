from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import os
import json
from datetime import datetime
from opencv_logic import process_frame as measure_process_frame

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

client_processing_options = {}
client_pose_instances = {}
client_measurement_state = {}


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

    processed_frame, angle = process_frame(frame, processing_type, pose_instance)

    # Measurement logic
    measurement_state = client_measurement_state.get(request.sid)
    if measurement_state and measurement_state["measurement_started"]:
        if angle is not None:
            current_max_angle = measurement_state["max_angle"]
            if current_max_angle is None or angle > current_max_angle:
                # Update max_angle and store the frame
                measurement_state["max_angle"] = angle
                measurement_state["max_angle_frame"] = processed_frame.copy()
            elif current_max_angle - angle >= 20:
                # Angle decreased by 20 or more, end measurement
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

    # Encode frame as JPEG with maximum quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode(".jpg", processed_frame, encode_param)
    frame_data = buffer.tobytes()

    socketio.emit("processed_frame", frame_data, to=request.sid)


@socketio.on("begin_measurement")
def handle_begin_measurement():
    # Initialize measurement state for the client
    client_measurement_state[request.sid] = {
        "measurement_started": True,
        "max_angle": None,
        "max_angle_frame": None,
        "processing_type": client_processing_options.get(request.sid, "default"),
        "joint_name": client_processing_options.get(request.sid, "default"),
    }
    print(f"Measurement started for client {request.sid}")


def process_frame(frame, processing_type, pose_instance):
    if processing_type in ["knee", "elbow", "shoulder"]:
        frame, angle = measure_process_frame(frame, processing_type, pose_instance)
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
    return frame, angle


def save_measurement(measurement_state, client_id):
    max_angle = measurement_state["max_angle"]
    max_angle_frame = measurement_state["max_angle_frame"]
    joint_name = measurement_state["joint_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{joint_name}_{timestamp}_{client_id}"

    # Ensure measurements directory exists
    if not os.path.exists("measurements"):
        os.makedirs("measurements")

    # Save the image
    image_filename = os.path.join("measurements", f"{filename_base}.jpg")
    cv2.imwrite(image_filename, max_angle_frame)

    # Save the measurement result as JSON
    measurement_data = {
        "joint_name": joint_name,
        "max_angle": max_angle,
        "timestamp": timestamp,
    }
    json_filename = os.path.join("measurements", f"{filename_base}.json")
    with open(json_filename, "w") as f:
        json.dump(measurement_data, f)

    print(f"Measurement saved for client {client_id}: angle {max_angle} at {timestamp}")


# 'assert status_set is not None, "write() before start_response"' error is due
# to running locally. Would not appear on production.
# https://github.com/miguelgrinberg/flask-sock/issues/27
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)
