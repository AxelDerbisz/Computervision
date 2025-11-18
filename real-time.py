import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# =========================================
# 1. CONFIGURATION & SETUP
# =========================================
MODEL_PATH = 'driver_behavior_model_optimized.keras'
INPUT_SIZE = (224, 224)

# Define class labels exactly as they were in your training notebook
CLASS_NAMES = {
    0: 'DangerousDriving',
    1: 'Distracted',
    2: 'Drinking',
    3: 'SafeDriving',
    4: 'SleepyDriving',
    5: 'Yawn'
}

# Classes that trigger an alert
ALERT_CLASSES = ['DangerousDriving', 'Distracted', 'Drinking', 'SleepyDriving', 'Yawn']

# Load the trained model
print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# =========================================
# 2. PREPROCESSING FUNCTION
# =========================================
def preprocess_frame(frame):
    """
    Preprocesses the frame to match the training notebook:
    1. Convert BGR to RGB
    2. Resize to 224x224
    3. Rescale pixel values (1./255)
    """
    # Convert BGR (OpenCV standard) to RGB (Model standard)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    resized = cv2.resize(rgb_frame, INPUT_SIZE)
    
    # Normalize to [0, 1] as done in ImageDataGenerator(rescale=1./255)
    normalized = resized.astype('float32') / 255.0
    
    # Expand dimensions to create a batch of 1: (1, 224, 224, 3)
    batch_input = np.expand_dims(normalized, axis=0)
    
    return batch_input

# =========================================
# 3. MAIN LOOP
# =========================================
def main():
    # Open Webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (optional, for performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # --- A. MediaPipe Face Detection Logic ---
            # To improve performance, optionally mark the image as not writeable to pass by reference
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            # Draw face bounding box
            frame.flags.writeable = True
            face_detected = False
            
            if results.detections:
                face_detected = True
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            # --- B. MobileNetV2 Behavior Detection Logic ---
            # Preprocess the FULL frame (Model needs context: hands, phone, etc.)
            input_tensor = preprocess_frame(frame)
            
            # Predict
            start_time = time.time()
            predictions = model.predict(input_tensor, verbose=0)
            end_time = time.time()
            
            # Process results
            predicted_class_id = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_id]
            label = CLASS_NAMES[predicted_class_id]

            # --- C. Visualization ---
            FPS = 1.0 / (end_time - start_time)
            
            # Determine color based on safety
            # Green for Safe, Red for others
            color = (0, 255, 0) if label == 'SafeDriving' else (0, 0, 255)
            
            # Display Text
            text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)
            
            # Display Warning if Face not detected but car is moving (simulation)
            if not face_detected:
                cv2.putText(frame, "WARNING: NO FACE DETECTED", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Visual Alert for Distraction
            if label in ALERT_CLASSES and confidence > 0.7:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
                cv2.putText(frame, "ALERT!", (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            # Show the output
            cv2.imshow('Driver Inattention Detection', frame)

            # Press 'q' to exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()