import cv2
import numpy as np
import mediapipe as mp
import warnings

# ignore that specific protobuf warning
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)

mp_drawing = mp.solutions.drawing_utils

# maps processing modes to the landmarks needed for angle calculation
MODE_TO_LANDMARKS = {
    "knee": ["HIP", "KNEE", "ANKLE"],
    "elbow": ["SHOULDER", "ELBOW", "WRIST"],
    "shoulder": ["ELBOW", "SHOULDER", "HIP"],
    "elbow-horizontal": ["WRIST", "ELBOW"],  # special handling for horizontal angle
}

# toggle confidence threshold check on or off
USE_CONFIDENCE_THRESHOLD = True


def calculateAngle(a, b, c):
    """Calculate the angle between three points (a, b, c), where b is the vertex."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # calculate cosine angle using dot product
    # add epsilon to avoid division by zero
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    # clip to [-1, 1] to handle potential float errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # convert cosine to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def getLandmarkPosition(landmarks, landmarkName, frame):
    """Get pixel coordinates for a specific landmark."""
    index = mp.solutions.pose.PoseLandmark[landmarkName].value
    return [
        landmarks[index].x * frame.shape[1],  # scale x coordinate
        landmarks[index].y * frame.shape[0],  # scale y coordinate
    ]


def drawJointVisualizations(image, pos1, pos2, pos3):
    """
    Draw lines, circles, and a filled arc for the angle visualization
    between pos1, pos2 (vertex), and pos3.
    """
    pos1_int = tuple(np.int32(pos1))
    pos2_int = tuple(np.int32(pos2))
    pos3_int = tuple(np.int32(pos3))

    # draw lines between joints
    cv2.line(image, pos1_int, pos2_int, (0, 0, 0), 3)
    cv2.line(image, pos2_int, pos3_int, (0, 0, 0), 3)

    # draw circles on joints
    cv2.circle(image, pos1_int, 10, (255, 0, 0), cv2.FILLED)  # endpoint 1
    cv2.circle(image, pos2_int, 10, (0, 255, 255), cv2.FILLED)  # vertex
    cv2.circle(image, pos3_int, 10, (255, 0, 0), cv2.FILLED)  # endpoint 2

    # --- draw the angle arc ---
    ba = np.array(pos1) - np.array(pos2)
    bc = np.array(pos3) - np.array(pos2)

    # calculate angles relative to x-axis for drawing arc
    angle_ba = np.degrees(np.arctan2(ba[1], ba[0])) % 360
    angle_bc = np.degrees(np.arctan2(bc[1], bc[0])) % 360

    # get start/end angles for the shortest arc path
    start_angle = angle_ba
    end_angle = angle_bc
    angle_between = (end_angle - start_angle) % 360
    if angle_between > 180:
        angle_between = 360 - angle_between
        start_angle, end_angle = end_angle, start_angle  # swap if needed

    # arc radius based on shorter connecting line
    length_ba = np.linalg.norm(ba)
    length_bc = np.linalg.norm(bc)
    # make sure radius is at least 1 pixel
    arc_radius = max(1, int(min(length_ba, length_bc) * 0.5))

    # draw arc outline
    center = pos2_int
    axes = (arc_radius, arc_radius)
    cv2.ellipse(
        image,
        center,
        axes,
        angle=0,  # ellipse rotation
        startAngle=start_angle,
        endAngle=start_angle + angle_between,
        color=(0, 0, 0),
        thickness=2,
    )

    # --- fill the arc area ---
    if arc_radius > 0 and angle_between > 0:  # check parameters are valid
        num_points = 50  # points to approximate arc for fill
        angle_range = np.linspace(start_angle, start_angle + angle_between, num_points)
        arc_points = []
        for angle in angle_range:
            theta = np.radians(angle)
            x = center[0] + arc_radius * np.cos(theta)
            y = center[1] + arc_radius * np.sin(theta)
            arc_points.append((int(x), int(y)))

        # define polygon vertices: center -> arc start -> arc points -> arc end -> center
        line1_point = (
            center[0] + arc_radius * np.cos(np.radians(start_angle)),
            center[1] + arc_radius * np.sin(np.radians(start_angle)),
        )
        # use the calculated end angle
        end_angle_rad = np.radians(start_angle + angle_between)
        line2_point = (
            center[0] + arc_radius * np.cos(end_angle_rad),
            center[1] + arc_radius * np.sin(end_angle_rad),
        )
        polygon_points = (
            [center]
            + [tuple(np.int32(line1_point))]
            + arc_points
            + [tuple(np.int32(line2_point))]
        )
        polygon_points = np.array([polygon_points], dtype=np.int32)

        # create overlay for transparency
        overlay = image.copy()
        cv2.fillPoly(overlay, polygon_points, color=(0, 0, 255))  # fill red

        # blend overlay with original image
        alpha = 0.3  # transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def drawTextAtPoint(image, text, point):
    """Draw text near a specified point."""
    cv2.putText(
        image,
        text,
        tuple(np.int32(point)),  # point needs to be integer tuple
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # font scale
        (0, 255, 255),  # color (yellow)
        2,  # thickness
        cv2.LINE_AA,  # line type
    )


def process_frame(frame, processing_type, pose_instance):
    """
    Process one frame: detect pose, calculate angles based on type, draw visualizations.

    Args:
        frame: Input frame (BGR format).
        processing_type (str): Angle type ('knee', 'elbow', etc).
        pose_instance: Initialized MediaPipe Pose object.

    Returns:
        tuple: (processed_image, angle, confidence)
               - processed_image: Frame with visualizations.
               - angle: Calculated angle in degrees (or None).
               - confidence: Minimum visibility of relevant landmarks (or None).
    """
    angle = None
    confidence = None
    confidence_threshold = 0.3  # minimum visibility score to trust landmarks

    if processing_type not in MODE_TO_LANDMARKS:
        # if not calculating angle, just show default message
        cv2.putText(
            frame,
            "Streaming...",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        return frame, angle, confidence

    # convert bgr to rgb for mediapipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False  # performance boost

    # detect pose landmarks
    results = pose_instance.process(image_rgb)

    # convert rgb back to bgr for opencv drawing
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # get landmark names needed for this mode (e.g., RIGHT_HIP)
            landmarkNames = [
                f"RIGHT_{bodyPart}" for bodyPart in MODE_TO_LANDMARKS[processing_type]
            ]

            positions = []
            confidences = []
            valid_landmarks = True
            # get pixel positions and confidence scores
            for landmarkName in landmarkNames:
                try:
                    position = getLandmarkPosition(landmarks, landmarkName, frame)
                    positions.append(position)
                    index = mp.solutions.pose.PoseLandmark[landmarkName].value
                    visibility = landmarks[
                        index
                    ].visibility  # mediapipe calls confidence 'visibility'
                    confidences.append(visibility)
                except IndexError:
                    valid_landmarks = False
                    # print(f"warning: landmark {landmarkName} not found.")
                    break  # stop if a required landmark is missing

            if not valid_landmarks:
                # message if essential landmarks are missing
                cv2.putText(
                    image,
                    f"Landmarks missing for {processing_type}",
                    (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                return image, None, None  # return early

            confidence = min(
                confidences
            )  # overall confidence is minimum of relevant landmarks

            # special case: 'elbow-horizontal', need a 3rd point offset horizontally
            if processing_type == "elbow-horizontal":
                offset = -200  # horizontal offset in pixels
                # point 3 is offset from point 2 (elbow)
                # original logic: elbow_x - offset = elbow_x - (-200) = elbow_x + 200 (point to the RIGHT)
                point_horizontal_offset = [
                    positions[1][0] - offset,
                    positions[1][1],
                ]  # reverted to original calculation
                positions.append(point_horizontal_offset)  # add as 3rd point

            # check confidence threshold (if enabled)
            if not USE_CONFIDENCE_THRESHOLD or all(
                conf >= confidence_threshold for conf in confidences
            ):
                # need exactly 3 points for angle calculation
                if len(positions) == 3:
                    # calculate and visualize angle
                    angle = calculateAngle(*positions)
                    drawJointVisualizations(image, *positions)  # pass all 3 points
                    drawTextAtPoint(
                        image,
                        f"{processing_type.capitalize()} Angle: {int(angle)} deg",
                        positions[1],  # display text near vertex
                    )
                else:
                    # safeguard - should not happen with current logic
                    cv2.putText(
                        image,
                        f"Incorrect number of points for {processing_type} angle",
                        (25, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                # message if confidence too low
                required_parts = ", ".join(
                    [bp.lower() for bp in MODE_TO_LANDMARKS[processing_type]]
                )
                cv2.putText(
                    image,
                    f"Low confidence - ensure {required_parts} are in frame",
                    (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,  # smaller font
                    (0, 0, 255), 
                    2,
                    cv2.LINE_AA,
                )
        else:
            # message if no pose detected
            cv2.putText(
                image,
                "No pose detected - move into frame",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    except Exception as e:
        # general error during processing or drawing
        # print(f"error processing frame: {e}") # uncomment for debugging
        cv2.putText(
            image,
            "Processing error occurred",
            (25, 80),  # lower position on screen
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return image, angle, confidence
