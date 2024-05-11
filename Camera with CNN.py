import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the defect detection model
def load_defect_detection_model(model_path):
    try:
        defect_detection_model = load_model(model_path)
        return defect_detection_model
    except Exception as e:
        print("Error loading defect detection model:", str(e))
        return None

# Function to preprocess image for defect detection
def preprocess_image(image):
    img_height, img_width = 150, 150  # Define image dimensions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    resized_img = cv2.resize(masked_img, (img_height, img_width))
    return resized_img

# Function to perform defect detection
# Function to perform defect detection
def detect_defect(image, model):
    preprocessed_image = preprocess_image(image)
    defect_probability = model.predict(np.expand_dims(preprocessed_image, axis=0))[0][0]
    print("Defect Probability:", defect_probability)
    return defect_probability > 0.5, defect_probability  # Return True if defect, False otherwise


# Camera code for defect detection
def defect_detection_camera(model):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break
        
        # Detect defects in the frame
        is_defect, defect_probability = detect_defect(frame, model)
        
        # Draw bounding box and defect probability on frame
        label = 'Defect' if is_defect else 'No Defect'
        color = (0, 0, 255) if is_defect else (0, 255, 0)
        cv2.putText(frame, f'{label} ({defect_probability:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display the frame
        cv2.imshow('Defect Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Load defect detection model
model_path = r'C:\Users\pixel\PycharmProjects\pythonProject\Arsitektur IoT\Computer Vision\defect_classification_model.keras'
defect_detection_model = load_defect_detection_model(model_path)

# Run defect detection using camera
if defect_detection_model:
    defect_detection_camera(defect_detection_model)
else:
    print("Error: Defect detection model could not be loaded")
