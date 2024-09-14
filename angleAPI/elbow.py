import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# calc angle between three points
def calculate_angle(a, b, c):
    # calculates angle at point 'b' formed by line segments ab and cb
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def main():
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("ignoring empty camera frame.")
                continue

            # Recolor image to RGB as MediaPipe uses RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for shoulder, elbow, and wrist
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

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                cv2.putText(image, str(int(angle)),
                            tuple(np.int32(elbow)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                            )

                cv2.line(image, tuple(np.int32(shoulder)), tuple(np.int32(elbow)), (0, 255, 0), 3)
                cv2.line(image, tuple(np.int32(elbow)), tuple(np.int32(wrist)), (0, 255, 0), 3)

                cv2.circle(image, tuple(np.int32(shoulder)), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(elbow)), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(wrist)), 10, (0, 0, 255), cv2.FILLED)

            except:
                pass

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            cv2.imshow('Elbow Angle Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()