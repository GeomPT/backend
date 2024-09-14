import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by the line segments ab and cb.
    
    Parameters:
    - a, b, c: Each a list or array with two elements representing x and y coordinates.
    
    Returns:
    - angle in degrees.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    # Calculate the vectors
    ba = a - b
    bc = c - b
    
    # Calculate the cosine of the angle using the dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    
    # Ensure the cosine value is within the valid range to avoid NaNs due to floating point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate the angle in radians and then convert to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def main():
    # Initialize webcam video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    # Set video width and height (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
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

                # Get coordinates for Elbow Angle (Right Side)
                shoulder_r = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]
                ]
                elbow_r = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]
                ]
                wrist_r = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]
                ]

                # Calculate Elbow Angle
                elbow_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)

                # Get coordinates for Shoulder External Rotation (Right Side)
                hip_r = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]
                ]

                # Calculate Shoulder External Rotation Angle
                shoulder_ext_rot_angle = calculate_angle(elbow_r, shoulder_r, hip_r)

                # Visualize Elbow Angle
                cv2.putText(image, f'Elbow Angle: {int(elbow_angle)} deg',
                            tuple(np.int32(elbow_r)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Visualize Shoulder External Rotation Angle
                cv2.putText(image, f'Shoulder Ext Rot: {int(shoulder_ext_rot_angle)} deg',
                            tuple(np.int32(shoulder_r)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                            )

                # Draw lines for Elbow Angle
                cv2.line(image, tuple(np.int32(shoulder_r)), tuple(np.int32(elbow_r)), (255, 255, 255), 3)
                cv2.line(image, tuple(np.int32(elbow_r)), tuple(np.int32(wrist_r)), (255, 255, 255), 3)

                # Draw lines for Shoulder External Rotation
                cv2.line(image, tuple(np.int32(hip_r)), tuple(np.int32(shoulder_r)), (0, 255, 0), 3)
                cv2.line(image, tuple(np.int32(shoulder_r)), tuple(np.int32(elbow_r)), (0, 255, 0), 3)

                # Draw circles on the joints
                # Elbow Angle Joints
                cv2.circle(image, tuple(np.int32(shoulder_r)), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(elbow_r)), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(wrist_r)), 10, (255, 0, 0), cv2.FILLED)

                # Shoulder External Rotation Joints
                cv2.circle(image, tuple(np.int32(hip_r)), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, tuple(np.int32(shoulder_r)), 10, (0, 255, 0), cv2.FILLED)

            except:
                pass

            # Render pose landmarks on the image
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Display the resulting image
            cv2.imshow('Elbow and Shoulder Angle Detection', image)

            # Exit condition
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
