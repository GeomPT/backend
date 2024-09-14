# test for shoulder knee and elbow
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def main():
    cap = cv2.VideoCapture(0)
    mode = 'knee'
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = Trues
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                if mode == 'knee':
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame.shape[1],
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame.shape[0]]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame.shape[0]]
                    angle = calculate_angle(hip, knee, ankle)
                    cv2.putText(image, f'Knee Angle: {int(angle)} deg',
                                tuple(np.int32(knee)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(np.int32(hip)), tuple(np.int32(knee)), (255, 255, 0), 3)
                    cv2.line(image, tuple(np.int32(knee)), tuple(np.int32(ankle)), (255, 255, 0), 3)
                    cv2.circle(image, tuple(np.int32(hip)), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(image, tuple(np.int32(knee)), 10, (0, 255, 255), cv2.FILLED)
                    cv2.circle(image, tuple(np.int32(ankle)), 10, (255, 0, 0), cv2.FILLED)
                elif mode == 'elbow':
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    cv2.putText(image, f'Elbow Angle: {int(angle)} deg',
                                tuple(np.int32(elbow)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(np.int32(shoulder)), tuple(np.int32(elbow)), (255, 255, 255), 3)
                    cv2.line(image, tuple(np.int32(elbow)), tuple(np.int32(wrist)), (255, 255, 255), 3)
                    cv2.circle(image, tuple(np.int32(shoulder)), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(image, tuple(np.int32(elbow)), 10, (255, 255, 255), cv2.FILLED)
                    cv2.circle(image, tuple(np.int32(wrist)), 10, (255, 0, 0), cv2.FILLED)
                elif mode == 'shoulder':
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]
                    angle = calculate_angle(elbow, shoulder, hip)
                    cv2.putText(image, f'Shoulder Ext Rot: {int(angle)} deg',
                                tuple(np.int32(shoulder)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, tuple(np.int32(elbow)), tuple(np.int32(shoulder)), (0, 255, 0), 3)
                    cv2.line(image, tuple(np.int32(shoulder)), tuple(np.int32(hip)), (0, 255, 0), 3)
                    cv2.circle(image, tuple(np.int32(elbow)), 10, (255, 255, 255), cv2.FILLED)
                    cv2.circle(image, tuple(np.int32(shoulder)), 10, (0, 255, 0), cv2.FILLED)
                    cv2.circle(image, tuple(np.int32(hip)), 10, (255, 255, 255), cv2.FILLED)
            except:
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            cv2.putText(image, f'Mode: {mode.upper()}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(image, 'Press K for Knee, E for Elbow, S for Shoulder, Q to Quit', (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Joint Angle Detection', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('k'):
                mode = 'knee'
            elif key == ord('e'):
                mode = 'elbow'
            elif key == ord('s'):
                mode = 'shoulder'
            elif key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()