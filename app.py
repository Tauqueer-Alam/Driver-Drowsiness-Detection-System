import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mediapipe as mp
import tensorflow as tf
import math

app = Flask(__name__)
CORS(app)

print("Loading Models...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

from create_cnn_model import create_cnn
cnn_model = create_cnn()
cnn_model.load_weights('drowsiness_cnn_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
print("Models loaded successfully!")

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    coords = [(int(landmarks.landmark[i].x * img_w), int(landmarks.landmark[i].y * img_h)) for i in eye_indices]
    p2_p6 = euclidean_distance(coords[1], coords[5])
    p3_p5 = euclidean_distance(coords[2], coords[4])
    p1_p4 = euclidean_distance(coords[0], coords[3])
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear, coords

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    current_mode = data.get('mode', 'Custom Keras CNN (DL Image)')
    ear_thresh = float(data.get('ear_thresh', 0.25))

    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
         return jsonify({'error': 'Invalid image'}), 400

    # Flip horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)

    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = face_mesh.process(rgb_frame)

    is_drowsy_ear = False
    is_drowsy_haar = False
    is_drowsy_cnn = False

    cnn_status_str = "N/A"
    haar_eyes_found = 0
    avg_ear = 0.0

    # 1. EAR
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_ear, left_coords = eye_aspect_ratio(landmarks, LEFT_EYE, img_w, img_h)
        right_ear, right_coords = eye_aspect_ratio(landmarks, RIGHT_EYE, img_w, img_h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < ear_thresh:
            is_drowsy_ear = True

        if "EAR" in current_mode or "HYBRID" in current_mode:
            color = (0, 0, 255) if is_drowsy_ear else (0, 255, 0)
            cv2.polylines(frame, [np.array(left_coords, np.int32)], True, color, 2)
            cv2.polylines(frame, [np.array(right_coords, np.int32)], True, color, 2)

    # 2. Haar and 3. CNN
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (fx, fy, fw, fh) in faces:
        roi_gray = gray_frame[fy:fy+int(fh/2), fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        haar_eyes_found += len(eyes)
        if "Haar" in current_mode or "HYBRID" in current_mode:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
            
    if len(faces) > 0 and haar_eyes_found == 0:
        is_drowsy_haar = True

    if "CNN" in current_mode or "HYBRID" in current_mode:
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            _, left_coords = eye_aspect_ratio(landmarks, LEFT_EYE, img_w, img_h)
            _, right_coords = eye_aspect_ratio(landmarks, RIGHT_EYE, img_w, img_h)
            
            left_closed = False
            right_closed = False
            
            for coords, is_left in [(left_coords, True), (right_coords, False)]:
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                
                x_center = sum(x_coords) // len(coords)
                y_center = sum(y_coords) // len(coords)
                
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                # Model expects proportional square crop. Force a square bounding box.
                box_size = max(width, height) + 30 
                half_size = box_size // 2
                
                x_min = max(0, x_center - half_size)
                x_max = min(img_w, x_center + half_size)
                y_min = max(0, y_center - half_size)
                y_max = min(img_h, y_center + half_size)

                if x_max > x_min and y_max > y_min:
                    eye_img = gray_frame[y_min:y_max, x_min:x_max]
                    eye_img = cv2.resize(eye_img, (64, 64))
                    eye_img = eye_img / 255.0
                    eye_img = eye_img.reshape(1, 64, 64, 1)

                    prediction = cnn_model.predict(eye_img, verbose=0)
                    if np.argmax(prediction) == 0:
                        if is_left: left_closed = True
                        else: right_closed = True

            if left_closed and right_closed:
                is_drowsy_cnn = True
                cnn_status_str = "CLOSED"
            else:
                cnn_status_str = "OPEN"

            if current_mode == "Custom Keras CNN (DL Image)":
                color_left = (0, 0, 255) if left_closed else (0, 255, 0)
                color_right = (0, 0, 255) if right_closed else (0, 255, 0)
                cv2.polylines(frame, [np.array(left_coords, np.int32)], True, color_left, 2)
                cv2.polylines(frame, [np.array(right_coords, np.int32)], True, color_right, 2)
        else:
            cnn_status_str = "NO FACE"

    trigger_detected = False
    if current_mode == "MediaPipe EAR (DL Mapping)":
        trigger_detected = is_drowsy_ear
        status_to_return = "DROWSY" if is_drowsy_ear else "NORMAL"
    elif current_mode == "Haar Cascades (Classic)":
        trigger_detected = is_drowsy_haar if len(faces) > 0 else False
        status_to_return = "DROWSY" if trigger_detected else "NORMAL"
    elif current_mode == "Custom Keras CNN (DL Image)":
        trigger_detected = is_drowsy_cnn
        status_to_return = cnn_status_str
    elif current_mode == "HYBRID (Combine All)":
        if is_drowsy_ear and (is_drowsy_haar or is_drowsy_cnn):
            trigger_detected = True
        status_to_return = "DROWSY" if trigger_detected else "NORMAL"

    # Re-encode to send to browser
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': status_to_return,
        'trigger': trigger_detected,
        'ear': float(avg_ear),
        'image': 'data:image/jpeg;base64,' + encoded_image
    })

if __name__ == '__main__':
    # Run on all network interfaces (so mobile phone can connect via local IP)
    # Using 'adhoc' allows HTTPS which is strictly required by modern mobile browsers to access the camera!
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
