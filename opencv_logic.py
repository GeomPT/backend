import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the pose estimation model once
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

MODE_TO_LANDMARKS = {
    "knee": ["HIP", "KNEE", "ANKLE"],
    "elbow": ["SHOULDER", "ELBOW", "WRIST"],
    "shoulder": ["ELBOW", "SHOULDER", "HIP"],
}


def calculateAngle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def getLandmarkPosition(landmarks, landmarkName, frame):
    index = mp_pose.PoseLandmark[landmarkName].value
    return [
        landmarks[index].x * frame.shape[1],
        landmarks[index].y * frame.shape[0],
    ]


def drawJointVisualizations(image, pos1, pos2, pos3):
    cv2.line(image, tuple(np.int32(pos1)), tuple(np.int32(pos2)), (255, 255, 0), 3)
    cv2.line(image, tuple(np.int32(pos2)), tuple(np.int32(pos3)), (255, 255, 0), 3)
    cv2.circle(image, tuple(np.int32(pos1)), 10, (255, 0, 0), cv2.FILLED)
    cv2.circle(image, tuple(np.int32(pos2)), 10, (0, 255, 255), cv2.FILLED)
    cv2.circle(image, tuple(np.int32(pos3)), 10, (255, 0, 0), cv2.FILLED)


def drawTextAtPoint(image, text, point):
    cv2.putText(
        image,
        text,
        tuple(np.int32(point)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_frame(frame, processing_type):
    if processing_type not in MODE_TO_LANDMARKS:
        # Default processing: just return the frame with a message
        cv2.putText(
            frame, "Streaming...", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
        )
        return frame

    # Recolor image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Make pose detection
    results = pose.process(image_rgb)

    # Recolor back to BGR for OpenCV
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        landmarkNames = [
            f"RIGHT_{bodyPart}" for bodyPart in MODE_TO_LANDMARKS[processing_type]
        ]
        positions = [
            getLandmarkPosition(landmarks, landmarkName, frame)
            for landmarkName in landmarkNames
        ]

        angle = calculateAngle(*positions)
        drawJointVisualizations(image, *positions)
        drawTextAtPoint(
            image,
            f"{processing_type.capitalize()} Angle: {int(angle)} deg",
            positions[1],
        )
    except Exception as e:
        # If landmarks are not detected, pass without altering the image
        pass

    return image
