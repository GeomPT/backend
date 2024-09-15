from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
from opencv_logic import process_frame as measurement_process_frame

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

client_processing_options = {}
client_pose_instances = {}


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

    processed_frame = process_frame(frame, processing_type, pose_instance)

    # Encode frame as JPEG with maximum quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode(".jpg", processed_frame, encode_param)
    frame_data = buffer.tobytes()

    socketio.emit("processed_frame", frame_data, to=request.sid)


def process_frame(frame, processing_type, pose_instance):
    if processing_type in ["knee", "elbow", "shoulder"]:
        frame = measurement_process_frame(frame, processing_type, pose_instance)
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
    return frame

# 'assert status_set is not None, "write() before start_response"' error is due
# to running locally. Would not appear on production.
# https://github.com/miguelgrinberg/flask-sock/issues/27
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)
