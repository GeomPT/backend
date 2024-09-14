import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by the line segments ab and cb.

    Parameters:
    - a, b, c: Each a list or array with two elements representing x and y coordinates.

    Returns:
    - angle in degrees.
    """
    a = np.array(a)  # Point A (Hip)
    b = np.array(b)  # Point B (Knee)
    c = np.array(c)  # Point C (Ankle)

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

def main():
    # Initialize webcam video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    # Optional: Set video width and height for consistency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Recolor image to RGB as MediaPipe uses RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Recolor back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Select the side you want to monitor: Right or Left
                # For example, Right Knee
                # To monitor both, replicate the process for LEFT_KNEE
                side = 'RIGHT'  # Change to 'LEFT' if needed

                if side == 'RIGHT':
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
                else:
                    hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]
                    ]
                    knee = [
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]
                    ]
                    ankle = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]
                    ]

                # Calculate Knee Angle
                knee_angle = calculate_angle(hip, knee, ankle)

                # Visualize the knee angle
                cv2.putText(image, f'Knee Angle: {int(knee_angle)} deg',
                            tuple(np.int32(knee)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw lines for visualization
                cv2.line(image, tuple(np.int32(hip)), tuple(np.int32(knee)), (255, 255, 0), 3)
                cv2.line(image, tuple(np.int32(knee)), tuple(np.int32(ankle)), (255, 255, 0), 3)

                # Draw circles on the joints
                cv2.circle(image, tuple(np.int32(hip)), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(knee)), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(ankle)), 10, (255, 0, 0), cv2.FILLED)

            except Exception as e:
                # Uncomment the following line to debug
                # print(e)
                pass

            # Render pose landmarks on the image
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Display the resulting image
            cv2.imshow('Knee Angle Detection', image)

            # Exit condition: Press 'q' to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
