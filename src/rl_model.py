import tensorflow as tf
import pandas as pd
import numpy as np

# Read CSV data
df = pd.read_csv("src/posture_data.csv")

# Normalize coordinate values
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
def extract_features(row):
    vw, vh = safe_float(row["videoWidth"]), safe_float(row["videoHeight"])
    if vw is None or vh is None:
        return None
    
    # Normalize coordinates
    noseX, noseY = norm_x(row["nose_x"], vw), norm_y(row["nose_y"], vh)
    lshoX, lshoY = norm_x(row["left_shoulder_x"], vw), norm_y(row["left_shoulder_y"], vh)
    rshoX, rshoY = norm_x(row["right_shoulder_x"], vw), norm_y(row["right_shoulder_y"], vh)
    learX, learY = norm_x(row["left_ear_x"], vw), norm_y(row["left_ear_y"], vh)
    rearX, rearY = norm_x(row["right_ear_x"], vw), norm_y(row["right_ear_y"], vh)

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

# Process dataset
X, y = [], []
for _, row in df.iterrows():
    label = safe_float(row["label"])
    features = extract_features(row)
    if features and label is not None:
        X.append(features)
        y.append(label / 100)  # Normalize labels between 0-1

# Convert to NumPy arrays
X, y = np.array(X), np.array(y)

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(7,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mean_squared_error")

# Train Model
print("Training model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Save model
model.save("posture_model.h5")
print("Model saved as 'posture_model.h5'.")
