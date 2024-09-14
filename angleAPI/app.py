import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, request

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

current_mode = 'knee'

def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by the line segments ab and cb.

    Parameters:
    - a, b, c: Each a list or array with two elements representing x and y coordinates.

    Returns:
    - angle in degrees.
    """
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C

    # Vectors
    ba = a - b
    bc = c - b

    # Compute the cosine of the angle using the dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)

    # Clip the cosine to the valid range to prevent NaNs from floating point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angle in radians and convert to degrees
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

def generate_frames():
    global current_mode
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        results = pose.process(image)

        # Recolor back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            if current_mode == 'knee':
                hip = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]
                ]
                knee = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame.shape[0]
                ]
                ankle = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame.shape[0]
                ]

                angle = calculate_angle(hip, knee, ankle)

                # Visualize the knee angle
                cv2.putText(image, f'Knee Angle: {int(angle)} deg',
                            tuple(np.int32(knee)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw lines for visualization
                cv2.line(image, tuple(np.int32(hip)), tuple(np.int32(knee)), (255, 255, 0), 3)
                cv2.line(image, tuple(np.int32(knee)), tuple(np.int32(ankle)), (255, 255, 0), 3)

                # Draw circles on the joints
                cv2.circle(image, tuple(np.int32(hip)), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(knee)), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(ankle)), 10, (255, 0, 0), cv2.FILLED)

            elif current_mode == 'elbow':
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]
                ]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize the elbow angle
                cv2.putText(image, f'Elbow Angle: {int(angle)} deg',
                            tuple(np.int32(elbow)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw lines for visualization
                cv2.line(image, tuple(np.int32(shoulder)), tuple(np.int32(elbow)), (255, 255, 255), 3)
                cv2.line(image, tuple(np.int32(elbow)), tuple(np.int32(wrist)), (255, 255, 255), 3)

                # Draw circles on the joints
                cv2.circle(image, tuple(np.int32(shoulder)), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(elbow)), 10, (255, 255, 255), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(wrist)), 10, (255, 0, 0), cv2.FILLED)

            elif current_mode == 'shoulder':
                elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]
                ]
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]
                ]
                hip = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]
                ]

                angle = calculate_angle(elbow, shoulder, hip)

                # Visualize the shoulder angle
                cv2.putText(image, f'Shoulder Angle: {int(angle)} deg',
                            tuple(np.int32(shoulder)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw lines for visualization
                cv2.line(image, tuple(np.int32(elbow)), tuple(np.int32(shoulder)), (0, 255, 0), 3)
                cv2.line(image, tuple(np.int32(shoulder)), tuple(np.int32(hip)), (0, 255, 0), 3)

                # Draw circles on the joints
                cv2.circle(image, tuple(np.int32(elbow)), 10, (255, 255, 255), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(shoulder)), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(hip)), 10, (255, 255, 255), cv2.FILLED)

        except Exception as e:
            # Uncomment the following line for debugging
            # print(e)
            pass

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<mode>')
def video_feed(mode):
    global current_mode
    if mode in ['knee', 'elbow', 'shoulder']:
        current_mode = mode
    else:
        current_mode = 'knee'  # Default mode
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
