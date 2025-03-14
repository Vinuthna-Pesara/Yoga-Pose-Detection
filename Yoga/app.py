import os
from flask import Flask, request, render_template, send_file
import cv2
import mediapipe as mp
import numpy as np
import joblib
from flask import jsonify
from scipy.spatial import distance  # Import this for distance calculation
from matplotlib import pyplot as plt
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the saved model
model = joblib.load('pose_classifier.pkl')

# Define MediaPipe Pose and labels
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
pose_labels = ['Chair', 'Cobra', 'Dog', 'No_Pose', 'Shoulderstand', 'Triangle', 'Tree', 'Warrior']


# Helper function to calculate distances
def calculate_distances(row):
    keypoints = ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
                 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
                 'LEFT_ANKLE', 'RIGHT_ANKLE']
    distances = []
    for i, kp1 in enumerate(keypoints):
        for kp2 in keypoints[i + 1:]:
            x1, y1 = row[f'{kp1}_x'], row[f'{kp1}_y']
            x2, y2 = row[f'{kp2}_x'], row[f'{kp2}_y']
            if not np.isnan([x1, y1, x2, y2]).any():
                distances.append(distance.euclidean((x1, y1), (x2, y2)))
            else:
                distances.append(0)  # Append 0 if any keypoint is missing
    return distances


# Helper function to calculate angles
def calculate_angles(row):
    angles = []
    keypoint_pairs = [("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP"),
                      ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE"),
                      ("LEFT_ELBOW", "LEFT_WRIST", "LEFT_SHOULDER")]
    for kp1, kp2, kp3 in keypoint_pairs:
        x1, y1 = row[f'{kp1}_x'], row[f'{kp1}_y']
        x2, y2 = row[f'{kp2}_x'], row[f'{kp2}_y']
        x3, y3 = row[f'{kp3}_x'], row[f'{kp3}_y']
        if not np.isnan([x1, y1, x2, y2, x3, y3]).any():
            v1 = np.array([x1 - x2, y1 - y2])
            v2 = np.array([x3 - x2, y3 - y2])
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip for numerical stability
            angles.append(np.degrees(angle))
        else:
            angles.append(0)  # Append 0 if any keypoint is missing
    return angles


def draw_keypoints_and_segments(image, landmarks, connections):
    h, w, _ = image.shape
    keypoints = {}
    for idx, lm in enumerate(landmarks.landmark):
        keypoints[mp_pose.PoseLandmark(idx).name] = (lm.x * w, lm.y * h, lm.visibility)

    # Draw keypoints
    for kp, values in keypoints.items():
        x, y, score = values
        if score > 0.5:  # Draw visible keypoints
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(image, kp, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw segments
    for kp1, kp2 in connections:
        if kp1 in keypoints and kp2 in keypoints:
            x1, y1, score1 = keypoints[kp1]
            x2, y2, score2 = keypoints[kp2]
            if score1 > 0.5 and score2 > 0.5:  # Draw if both keypoints are visible
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return image


# Helper function to process and predict pose
def predict_pose_from_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Pose
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return "No pose detected", image

    # Extract keypoints
    keypoints = {}
    h, w, _ = image.shape
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        keypoints[mp_pose.PoseLandmark(idx).name + '_x'] = lm.x * w
        keypoints[mp_pose.PoseLandmark(idx).name + '_y'] = lm.y * h

    # Draw pose on image
    image_with_keypoints = draw_keypoints_and_segments(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Predict pose using trained classifier
    try:
        distances = calculate_distances(keypoints)
        angles = calculate_angles(keypoints)
        features = np.hstack([distances, angles]).reshape(1, -1)
        class_no = model.predict(features)[0]
        predicted_pose = pose_labels[class_no]
    except Exception as e:
        predicted_pose = f"Error in prediction: {str(e)}"

    # Annotate image
    annotated_image = image.copy()
    for kp in keypoints:
        # Access the keypoint's x and y coordinates individually
        x = keypoints.get(kp + '_x', 0)
        y = keypoints.get(kp + '_y', 0)
        if x > 0 and y > 0:
            cv2.circle(annotated_image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(annotated_image, kp, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return predicted_pose, annotated_image


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process and predict
    pose_name, annotated_image = predict_pose_from_image(file_path)

    # Save the annotated image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file.filename.split('.')[0]}_output.jpg")
    cv2.imwrite(output_path, annotated_image)

    return jsonify({
        'pose': pose_name,
        'imagePath': f'uploads/{file.filename.split(".")[0]}_output.jpg'  # Send image path as relative URL
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

# Enable CORS for all routes
CORS(app)
if __name__ == '__main__':
    app.run(debug=True)
