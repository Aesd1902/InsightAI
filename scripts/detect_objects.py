import cv2
import numpy as np

def load_yolo_model(config_path, weights_path):
    """
    Load the YOLO model from configuration and weights files.
    
    Parameters:
    - config_path: Path to the YOLO configuration file.
    - weights_path: Path to the YOLO weights file.
    
    Returns:
    - net: Loaded YOLO network.
    """
    return cv2.dnn.readNetFromDarknet(config_path, weights_path)

def detect_objects(frame, net, output_layers, confidence_threshold=0.5):
    """
    Detect objects in a frame using the YOLO model.
    
    Parameters:
    - frame: Input image/frame.
    - net: YOLO network.
    - output_layers: Output layer names.
    - confidence_threshold: Minimum confidence threshold for detection.
    
    Returns:
    - boxes: List of bounding boxes.
    - confidences: List of confidences for each box.
    - class_ids: List of class IDs for each box.
    """
    height, width, _ = frame.shape
    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def draw_boxes(frame, boxes, confidences, class_ids, class_names):
    """
    Draw bounding boxes and labels on the frame with enhanced visibility.
    
    Parameters:
    - frame: Input image/frame.
    - boxes: List of bounding boxes.
    - confidences: List of confidences for each box.
    - class_ids: List of class IDs for each box.
    - class_names: List of class names corresponding to class IDs.
    """
    # Get indices of objects with non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    if len(indices) > 0:
        for i in indices.flatten():  # Flatten the index tuple
            box = boxes[i]
            x, y, w, h = box
            
            # Define label with the class name and confidence score
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            
            # Draw bounding box with thicker border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Adjust font scale and thickness for better readability
            font_scale = 0.9  # Medium font size
            font_thickness = 2
            
            # Draw the text label with larger font
            cv2.putText(
                frame, 
                label, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 0), 
                font_thickness
            )

if __name__ == "__main__":
    # Load YOLO model
    config_path = "C:/Users/Uday Alugolu/OneDrive/Desktop/image_classification_video/models/yolov3.cfg"
    weights_path = "C:/Users/Uday Alugolu/OneDrive/Desktop/image_classification_video/models/yolov3.weights"
    
    net = load_yolo_model(config_path, weights_path)

    # Load class names
    try:
        with open("C:/Users/Uday Alugolu/OneDrive/Desktop/image_classification_video/models/coco.names", "r") as f:
            class_names = f.read().strip().split("\n")
    except FileNotFoundError:
        print("Error: coco.names file not found.")
        exit()

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Open video capture
    video_path = "data/video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Set up the display window in fullscreen
    cv2.namedWindow("YOLO Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in the frame
        boxes, confidences, class_ids = detect_objects(frame, net, output_layers)
        
        # Draw bounding boxes on the frame
        draw_boxes(frame, boxes, confidences, class_ids, class_names)
        
        # Show the result in fullscreen
        cv2.imshow("YOLO Detection", frame)
        
        # Press 'q' to exit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Print OpenCV version
    print("OpenCV version:", cv2.__version__)
