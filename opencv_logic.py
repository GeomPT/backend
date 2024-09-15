import cv2
import numpy as np
import mediapipe as mp
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)

mp_drawing = mp.solutions.drawing_utils

MODE_TO_LANDMARKS = {
    "knee": ["HIP", "KNEE", "ANKLE"],
    "elbow": ["SHOULDER", "ELBOW", "WRIST"],
    "shoulder": ["ELBOW", "SHOULDER", "HIP"],
}

# Boolean to toggle confidence threshold check
USE_CONFIDENCE_THRESHOLD = True


def calculateAngle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B (vertex)
    c = np.array(c)  # Point C

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def getLandmarkPosition(landmarks, landmarkName, frame):
    index = mp.solutions.pose.PoseLandmark[landmarkName].value
    return [
        landmarks[index].x * frame.shape[1],
        landmarks[index].y * frame.shape[0],
    ]


def drawJointVisualizations(image, pos1, pos2, pos3):
    """
    Draws lines, circles, and an arc to visualize the angle calculation.
    pos1 - pos2 - pos3
    Also fills the area between the arc and lines with a translucent red.
    """
    # Convert positions to integer tuples
    pos1 = tuple(np.int32(pos1))
    pos2 = tuple(np.int32(pos2))
    pos3 = tuple(np.int32(pos3))

    # Draw lines for visualization
    cv2.line(image, pos1, pos2, (0, 0, 0), 3)
    cv2.line(image, pos2, pos3, (0, 0, 0), 3)

    # Draw circles on the joints
    cv2.circle(image, pos1, 10, (255, 0, 0), cv2.FILLED)
    cv2.circle(image, pos2, 10, (0, 255, 255), cv2.FILLED)
    cv2.circle(image, pos3, 10, (255, 0, 0), cv2.FILLED)

    # Draw arc to visualize the angle
    # Calculate vectors
    ba = np.array(pos1) - np.array(pos2)
    bc = np.array(pos3) - np.array(pos2)

    # Calculate the angles of the vectors relative to the x-axis
    angle_ba = np.degrees(np.arctan2(ba[1], ba[0]))
    angle_bc = np.degrees(np.arctan2(bc[1], bc[0]))

    # Ensure the angles are in [0, 360)
    angle_ba = angle_ba % 360
    angle_bc = angle_bc % 360

    # Determine start and end angles for the arc
    start_angle = angle_ba
    end_angle = angle_bc

    # Calculate the smallest angle between the two vectors
    angle_between = (end_angle - start_angle) % 360
    if angle_between > 180:
        angle_between = 360 - angle_between
        start_angle, end_angle = end_angle, start_angle

    # Define the radius of the arc (relative to shorter line)
    length_ba = np.linalg.norm(ba)
    length_bc = np.linalg.norm(bc)
    arc_radius = int(min(length_ba, length_bc) * 0.5)

    # Draw the arc
    center = pos2
    axes = (arc_radius, arc_radius)
    cv2.ellipse(
        image,
        center,
        axes,
        0,
        start_angle,
        start_angle + angle_between,
        (0, 0, 0),
        2,
    )

    # Create points along the arc
    num_points = 50  # More points for a smoother arc
    angle_range = np.linspace(start_angle, start_angle + angle_between, num_points)
    arc_points = []
    for angle in angle_range:
        theta = np.radians(angle)
        x = center[0] + arc_radius * np.cos(theta)
        y = center[1] + arc_radius * np.sin(theta)
        arc_points.append((int(x), int(y)))

    # Points along the lines (from center to arc endpoints)
    line1_point = (
        center[0] + arc_radius * np.cos(np.radians(start_angle)),
        center[1] + arc_radius * np.sin(np.radians(start_angle)),
    )
    line2_point = (
        center[0] + arc_radius * np.cos(np.radians(end_angle)),
        center[1] + arc_radius * np.sin(np.radians(end_angle)),
    )

    # Build the polygon points
    polygon_points = [center, line1_point] + arc_points + [line2_point, center]
    polygon_points = np.array([polygon_points], dtype=np.int32)

    # Create an overlay to draw the translucent area
    overlay = image.copy()
    cv2.fillPoly(overlay, polygon_points, color=(0, 0, 255))

    # Blend the overlay with the original image
    alpha = 0.3  # Transparency factor (0: transparent, 1: opaque)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


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


def process_frame(frame, processing_type, pose_instance):
    angle = None  # Initialize angle to None
    confidence = None  # Initialize confidence to None
    confidence_threshold = 0.5  # Adjust this threshold as needed

    if processing_type not in MODE_TO_LANDMARKS:
        # Default processing: just return the frame with a message
        cv2.putText(
            frame, "Streaming...", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
        )
        return frame, angle, confidence  # Return angle as None

    # Recolor image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Process the image using the provided pose_instance
    results = pose_instance.process(image_rgb)

    # Recolor back to BGR for OpenCV
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        landmarkNames = [
            f"RIGHT_{bodyPart}" for bodyPart in MODE_TO_LANDMARKS[processing_type]
        ]
        positions = []
        confidences = []
        for landmarkName in landmarkNames:
            position = getLandmarkPosition(landmarks, landmarkName, frame)
            positions.append(position)
            index = mp.solutions.pose.PoseLandmark[landmarkName].value
            visibility = landmarks[index].visibility
            confidences.append(visibility)
        confidence = min(confidences)  # Use the minimum confidence among landmarks

        # Check if all confidences are above the threshold
        if not USE_CONFIDENCE_THRESHOLD or all(
            conf >= confidence_threshold for conf in confidences
        ):
            angle = calculateAngle(*positions)
            drawJointVisualizations(image, *positions)
            drawTextAtPoint(
                image,
                f"{processing_type.capitalize()} Angle: {int(angle)} deg",
                positions[1],
            )
        else:
            # Low confidence in landmarks
            cv2.putText(
                image,
                "Please make sure the camera can see all of you clearly",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
    except Exception as e:
        # Landmarks not detected
        cv2.putText(
            image,
            "Please move into frame",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

    return image, angle, confidence
