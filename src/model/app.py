from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model("src/model/posture_model.h5")

# Normalize coordinate values and helper functions
def safe_float(val):
    try:
        return float(val)
    except ValueError:
        return None

def norm_x(col, vw):
    v = safe_float(col)
    return v / vw if v is not None else None

def norm_y(col, vh):
    v = safe_float(col)
    return v / vh if v is not None else None

def distance2D(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def angleABC(Ax, Ay, Bx, By, Cx, Cy):
    ABx, ABy = Ax - Bx, Ay - By
    CBx, CBy = Cx - Bx, Cy - By
    dot = ABx * CBx + ABy * CBy
    magAB, magCB = np.sqrt(ABx**2 + ABy**2), np.sqrt(CBx**2 + CBy**2)
    if magAB == 0 or magCB == 0:
        return 180.0
    cosTheta = np.clip(dot / (magAB * magCB), -1, 1)
    return np.degrees(np.arccos(cosTheta))

# Feature extraction function
def extract_features(data):
    vw, vh = safe_float(data["videoWidth"]), safe_float(data["videoHeight"])
    if vw is None or vh is None:
        return None
    
    # Normalize coordinates
    noseX, noseY = norm_x(data["nose_x"], vw), norm_y(data["nose_y"], vh)
    lshoX, lshoY = norm_x(data["left_shoulder_x"], vw), norm_y(data["left_shoulder_y"], vh)
    rshoX, rshoY = norm_x(data["right_shoulder_x"], vw), norm_y(data["right_shoulder_y"], vh)
    learX, learY = norm_x(data["left_ear_x"], vw), norm_y(data["left_ear_y"], vh)
    rearX, rearY = norm_x(data["right_ear_x"], vw), norm_y(data["right_ear_y"], vh)

    if None in [noseX, noseY, lshoX, lshoY, rshoX, rshoY, learX, learY, rearX, rearY]:
        return None

    mshoX, mshoY = (lshoX + rshoX) / 2, (lshoY + rshoY) / 2
    dist_nose_shoulders = distance2D(noseX, noseY, mshoX, mshoY)
    shoulder_width = distance2D(lshoX, lshoY, rshoX, rshoY)
    ratio_noseShoulders = dist_nose_shoulders / shoulder_width if shoulder_width > 0 else 0
    neck_tilt_angle = angleABC(learX, learY, noseX, noseY, rearX, rearY)
    dist_leftEar_nose = distance2D(learX, learY, noseX, noseY)
    dist_rightEar_nose = distance2D(rearX, rearY, noseX, noseY)
    angle_leftShoulder = angleABC(learX, learY, lshoX, lshoY, noseX, noseY)
    angle_rightShoulder = angleABC(rearX, rearY, rshoX, rshoY, noseX, noseY)

    return [
        dist_nose_shoulders,
        ratio_noseShoulders,
        neck_tilt_angle,
        dist_leftEar_nose,
        dist_rightEar_nose,
        angle_leftShoulder,
        angle_rightShoulder
    ]

# Define a POST route for posture prediction
@app.route('/predict_posture', methods=['POST'])
def predict_posture():
    # Get JSON data from request
    data = request.json
    
    # Extract features from the data
    features = extract_features(data)
    if features is None:
        return jsonify({"error": "Invalid data"}), 400

    # Prepare input for model
    features = np.array([features])

    # Predict posture
    pred = model.predict(features)[0][0]
    posture_score = int(pred * 100)

    # Return result
    result = {
        "posture_score": posture_score,
        "posture_label": "Good Posture" if posture_score > 85 else "Bad Posture"
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
