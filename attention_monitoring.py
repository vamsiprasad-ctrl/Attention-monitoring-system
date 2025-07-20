import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from threading import Thread
import winsound 
import csv
from datetime import datetime
import os

# Model
yawn_model = load_model("yawn_model.h5")


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


YAWN_THRESHOLD = 0.70  
YAWN_CONSEC_FRAMES = 17
ALERT_COOLDOWN = 5
NOT_ATTENTIVE_THRESHOLD = 6
DROWSY_EAR_THRESHOLD = 0.23
DROWSY_CONSEC_FRAMES = 30
HEAD_POSE_THRESHOLD = 20  # degrees
IMAGE_SAVE_DIR = "alerts"


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
HEAD_POSE_POINTS = [1, 33, 263, 61, 291, 199]  # Nose, eye corners, mouth corners, chin


yawn_counter = 0
last_alert_time = 0
last_attentive_time = time.time()
yawn_probs = []
drowsy_counter = 0
not_present_alerted = False


if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)


def log_alert(event_type, image_name=None):
    with open("alert_log.csv", mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, image_name or ""])


def sound_alert():
    winsound.Beep(900, 100)


def alert_event(event_type, frame):
    global last_alert_time
    if time.time() - last_alert_time > ALERT_COOLDOWN:
        print(f"ALERT: {event_type}")
        img_name = save_alert_image(event_type, frame)
        log_alert(event_type, img_name)
        sound_alert()
        last_alert_time = time.time()


def save_alert_image(event, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{event}_{timestamp}.jpg"
    path = os.path.join(IMAGE_SAVE_DIR, filename)
    cv2.imwrite(path, frame)
    return filename


def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return 0 if C == 0 else (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmark_coords):
    A = np.linalg.norm(np.array(landmark_coords[13]) - np.array(landmark_coords[14]))
    B = np.linalg.norm(np.array(landmark_coords[78]) - np.array(landmark_coords[82]))
    return 0 if B == 0 else A / B


mar_list = []
def smoothed_mar(current_mar, window=5):
    mar_list.append(current_mar)
    if len(mar_list) > window:
        mar_list.pop(0)
    return sum(mar_list) / len(mar_list)


def get_head_pose(landmarks, img_shape):
    image_points = np.array([landmarks[i] for i in HEAD_POSE_POINTS], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),      # Nose tip
        (-30.0, -30.0, -30.0), # Left eye
        (30.0, -30.0, -30.0),  # Right eye
        (-30.0, 30.0, -30.0),  # Left mouth
        (30.0, 30.0, -30.0),   # Right mouth
        (0.0, 60.0, -50.0)     # Chin
    ])

    focal_length = img_shape[1]
    center = (img_shape[1] // 2, img_shape[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None

    rot_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
        yaw = np.arctan2(-rot_matrix[2, 0], sy)
        roll = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    else:
        pitch = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
        yaw = np.arctan2(-rot_matrix[2, 0], sy)
        roll = 0

    return np.degrees([pitch, yaw, roll])


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not access webcam.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    face_found = False
    face_forward = True

    if results.multi_face_landmarks:
        face_found = True
        face_landmarks = results.multi_face_landmarks[0]
        landmark_coords = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

        
        try:
            head_pose = get_head_pose(landmark_coords, frame.shape)
            if head_pose is not None:
                pitch, yaw, roll = head_pose
                cv2.putText(frame, f"Pitch: {int(pitch)} Yaw: {int(yaw)} Roll: {int(roll)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if abs(pitch) > HEAD_POSE_THRESHOLD or abs(yaw) > HEAD_POSE_THRESHOLD:
                    Thread(target=alert_event, args=("HEAD_POSE_ABNORMAL", frame.copy())).start()
        except:
            pass

        
        try:
            left_eye_pts = [landmark_coords[i] for i in LEFT_EYE]
            right_eye_pts = [landmark_coords[i] for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0

            if ear < DROWSY_EAR_THRESHOLD:
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            if drowsy_counter >= DROWSY_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                Thread(target=alert_event, args=("DROWSINESS", frame.copy())).start()
        except:
            pass

        # Yawning
        try:
            mar = smoothed_mar(mouth_aspect_ratio(landmark_coords))
            if mar > 0.4:
                mouth_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mouth_img = cv2.resize(mouth_img, (64, 64)) / 255.0
                mouth_img = np.expand_dims(mouth_img, axis=0)
                yawn_prob = yawn_model.predict(mouth_img)[0][0]
                yawn_probs.append(yawn_prob)
                if len(yawn_probs) > 10:
                    yawn_probs.pop(0)

                if sum(yawn_probs) / len(yawn_probs) > YAWN_THRESHOLD:
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    cv2.putText(frame, "YAWNING", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    Thread(target=alert_event, args=("YAWNING", frame.copy())).start()
        except:
            pass

    else:
        if time.time() - last_attentive_time > NOT_ATTENTIVE_THRESHOLD and not not_present_alerted:
            Thread(target=alert_event, args=("NOT_PRESENT", frame.copy())).start()
            not_present_alerted = True

    cv2.imshow("Attentiveness Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
