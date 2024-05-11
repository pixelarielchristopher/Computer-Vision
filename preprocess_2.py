import cv2
import os

# Function to preprocess images in a folder
# Function to preprocess images in a folder and save only the cropped images
def preprocess_images_in_folder(input_folder, output_folder, canny_threshold1=100, canny_threshold2=200):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)
    
    # Counter for cropped images
    cropped_image_count = 0
    
    # Iterate through each file in the input folder
    for file in files:
        # Check if the file is an image file (assuming all files in the folder are images)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Read the image
            image_path = os.path.join(input_folder, file)
            image = cv2.imread(image_path)
            
            # Preprocess the image
            _, cropped_object = preprocess_frame(image, canny_threshold1, canny_threshold2)
            
            # Check if the cropped object is not empty and its size is above 400x400
            if cropped_object.size > 0 and cropped_object.shape[0] >= 350 and cropped_object.shape[1] >= 350:
                # Save the cropped object
                cropped_image_count += 1
                cropped_object_path = os.path.join(output_folder, f"cropped_image_{cropped_image_count}.jpg")
                cv2.imwrite(cropped_object_path, cropped_object)
                print(f"Processed and saved {file}")

    print(f"Total cropped images saved: {cropped_image_count}")


# Function to preprocess a single frame
def preprocess_frame(frame, canny_threshold1, canny_threshold2):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform Canny edge detection with the specified thresholds
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
    
    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if contours are found
    if contours:
        # Get the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the object region
        cropped_object = frame[y:y+h, x:x+w]
        
        return frame, cropped_object
    else:
        # If no contours are found, return the original frame and an empty image
        return frame, frame

# Path to the input folder containing images
input_folder = "C:/Users/pixel/PycharmProjects/pythonProject/Arsitektur IoT/Computer Vision/test"

# Path to the output folder where cropped images will be saved
output_folder = "C:/Users/pixel/PycharmProjects/pythonProject/Arsitektur IoT/Computer Vision/test/cropped_images"

# Call the function to preprocess images in the input folder and save only the cropped images
preprocess_images_in_folder(input_folder, output_folder)
