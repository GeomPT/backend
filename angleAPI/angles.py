import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

currentMode = "knee"

MODE_TO_LANDMARKS = {
    "knee": ["HIP", "KNEE", "ANKLE"],
    "elbow": ["SHOULDER", "ELBOW", "WRIST"],
    "shoulder": ["ELBOW", "SHOULDER", "HIP"],
    "elbow_horizontal": ["ELBOW", "WRIST"]  # Only two landmarks; third point is computed
}

def setCurrentMode(mode):
    global currentMode
    currentMode = mode

def calculateAngle(a, b, c):
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

def getLandmarkPosition(landmarks, landmarkName, frame):
    index = mp_pose.PoseLandmark[landmarkName].value
    return [
        landmarks[index].x * frame.shape[1],
        landmarks[index].y * frame.shape[0],
    ]

def drawJointVisualizations(image, pos1, pos2, pos3):
    """
    Draws lines and circles on the image to visualize the angle calculation.
    pos1 - pos2 - pos3
    """
    # Draw lines for visualization
    cv2.line(image, tuple(np.int32(pos1)), tuple(np.int32(pos2)), (255, 255, 0), 3)
    cv2.line(image, tuple(np.int32(pos2)), tuple(np.int32(pos3)), (255, 255, 0), 3)

    # Draw circles on the joints
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

def generateImageAndAngle(image, direction, landmarks, frame):
    """
    Generates visualizations and calculates angles based on the current mode.
    """
    if currentMode in ["knee", "elbow", "shoulder"]:
        landmarkNames = [
            f"{direction}_{bodyPart}" for bodyPart in MODE_TO_LANDMARKS[currentMode]
        ]
        positions = [
            getLandmarkPosition(landmarks, landmarkName, frame)
            for landmarkName in landmarkNames
        ]

        angle = calculateAngle(*positions)

        drawJointVisualizations(image, *positions)
        drawTextAtPoint(
            image, f"{currentMode.capitalize()} Angle: {int(angle)} deg", positions[1]
        )

    elif currentMode == "elbow_horizontal":
        # Get elbow and wrist positions
        elbow = getLandmarkPosition(landmarks, "RIGHT_ELBOW", frame)
        wrist = getLandmarkPosition(landmarks, "RIGHT_WRIST", frame)

        # Compute a point 50 pixels to the left of the elbow, aligned horizontally
        # Assuming 'left' is towards decreasing x-axis
        offset = -200
        point_left = [elbow[0] - offset, elbow[1]]

        angle = calculateAngle(wrist, elbow, point_left)

        # Draw lines and circles
        drawJointVisualizations(image, wrist, elbow, point_left)
        drawTextAtPoint(
            image, f"Elbow Horizontal Angle: {int(angle)} deg", elbow
        )

    return image, angle

def generateFrames():
    global currentMode
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

            generateImageAndAngle(image, "RIGHT", landmarks, frame)
        except Exception as e:
            pass

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

