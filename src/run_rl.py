import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load TensorFlow model
model = tf.keras.models.load_model("posture_model.h5")  # Change if needed

# Function to calculate 2D distance
def distance2D(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate angle between three points
def angleABC(Ax, Ay, Bx, By, Cx, Cy):
    ABx, ABy = Ax - Bx, Ay - By
    CBx, CBy = Cx - Bx, Cy - By

    dot = ABx * CBx + ABy * CBy
    magAB = np.sqrt(ABx ** 2 + ABy ** 2)
    magCB = np.sqrt(CBx ** 2 + CBy ** 2)
    if magAB == 0 or magCB == 0:
        return 180.0

    cosTheta = dot / (magAB * magCB)
    cosTheta = np.clip(cosTheta, -1, 1)
    return np.degrees(np.arccos(cosTheta))

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = frame.shape  # Get frame dimensions

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract required keypoints
        def get_landmark(name):
            return landmarks[mp_pose.PoseLandmark[name].value]

        nose = get_landmark("NOSE")
        left_shoulder = get_landmark("LEFT_SHOULDER")
        right_shoulder = get_landmark("RIGHT_SHOULDER")
        left_ear = get_landmark("LEFT_EAR")
        right_ear = get_landmark("RIGHT_EAR")

        # Normalize coordinates (VERY IMPORTANT for consistency with training)
        nose_x, nose_y = nose.x, nose.y
        lsho_x, lsho_y = left_shoulder.x, left_shoulder.y
        rsho_x, rsho_y = right_shoulder.x, right_shoulder.y
        lear_x, lear_y = left_ear.x, left_ear.y
        rear_x, rear_y = right_ear.x, right_ear.y

        # Compute features
        msho_x, msho_y = (lsho_x + rsho_x) / 2, (lsho_y + rsho_y) / 2
        dist_nose_shoulders = distance2D(nose_x, nose_y, msho_x, msho_y)
        shoulder_width = distance2D(lsho_x, lsho_y, rsho_x, rsho_y)
        ratio_noseShoulders = dist_nose_shoulders / shoulder_width if shoulder_width > 0 else 0
        neck_tilt_angle = angleABC(lear_x, lear_y, nose_x, nose_y, rear_x, rear_y)
        dist_leftEar_nose = distance2D(lear_x, lear_y, nose_x, nose_y)
        dist_rightEar_nose = distance2D(rear_x, rear_y, nose_x, nose_y)
        angle_leftShoulder = angleABC(lear_x, lear_y, lsho_x, lsho_y, nose_x, nose_y)
        angle_rightShoulder = angleABC(rear_x, rear_y, rsho_x, rsho_y, nose_x, nose_y)
        shoulder_width = distance2D(lsho_x, lsho_y, rsho_x, rsho_y)
        head_height = distance2D(nose_x, nose_y, msho_x, msho_y)
        ear_height_diff = abs(lear_y - rear_y) * h
        shoulder_height_diff = abs(lsho_y - rsho_y) * h


        # Debugging Prints
        print(f"Dist Nose-Shoulders: {dist_nose_shoulders:.2f}, Ratio: {ratio_noseShoulders:.2f}")
        print(f"Neck Tilt: {neck_tilt_angle:.2f}, Shoulder Angles: {angle_leftShoulder:.2f}, {angle_rightShoulder:.2f}")

        # Prepare input for model
        features = np.array([[dist_nose_shoulders, ratio_noseShoulders, neck_tilt_angle,
                              dist_leftEar_nose, dist_rightEar_nose, angle_leftShoulder, angle_rightShoulder]])

        # Predict posture
        pred = model.predict(features)[0][0]
        print("Raw Prediction:", pred)  # Debugging: Check if prediction is always 1.0

        # Convert prediction to posture score
        posture_score = int(pred * 100)
        label = "Good Posture" if posture_score > 85 else "Bad Posture"

        # Display label on frame
        cv2.putText(image, f"Posture: {label} ({posture_score}%)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Good Posture" else (0, 0, 255), 2)
        
        head_too_low_thresh = shoulder_width * 0.6
        head_too_far_thresh = shoulder_width * 1.2
        ear_level_thresh = 15  # Pixels threshold for level check
        shoulder_level_thresh = 20  # Pixels threshold for level check

        # Detect posture issues
        issues = []
        if ear_height_diff > ear_level_thresh:
            issues.append("Head Tilt Detected")
        if shoulder_height_diff > shoulder_level_thresh:
            issues.append("Shoulders Uneven")
        if head_height < head_too_low_thresh:
            issues.append("Head Too Low")
        if head_height > head_too_far_thresh:
            issues.append("Head Too Far Forward")

        # Display alerts
        for i, issue in enumerate(issues):
            cv2.putText(image, issue, (20, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show result
    cv2.imshow("Posture Analysis", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
