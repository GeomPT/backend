from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

client_processing_options = {}


@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    if request.sid in client_processing_options:
        del client_processing_options[request.sid]


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

    processed_frame = process_frame(frame, processing_type)

    # Encode frame as JPEG with maximum quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode(".jpg", processed_frame, encode_param)
    frame_data = buffer.tobytes()

    socketio.emit("processed_frame", frame_data, to=request.sid)


def process_frame(frame, processing_type):
    if processing_type == "grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif processing_type == "edge":
        frame = cv2.Canny(frame, 100, 200)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        cv2.putText(
            frame, "Streaming...", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
        )
    return frame


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000)
