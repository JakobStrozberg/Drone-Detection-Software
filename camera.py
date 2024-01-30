import torch  # pip install torch
import numpy as np # pip install numpy
import cv2  # pip install opencv-python
from time import time
from ultralytics import YOLO  # pip install ultralytics
import supervision as sv # pip install supervision
from ultralytics import RTDETR  # pip install ultralytics
from mss import mss 
import serial ###DELETE 
Serial = serial.Serial ### DELETE

class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        # Check if CUDA is available, otherwise use CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        # Initialize the YOLO model for object detection
        # Choose between the RTDETR model, the large YOLOv8 model, or the nano YOLOv8 model
        # The model files should be in the same directory as this script
        # The YOLO models are faster but less accurate than the RTDETR model
        # The nano YOLOv8 model is the fastest but least accurate of the three
        # The RTDETR model is the slowest but most accurate of the three
        
        # self.model = RTDETR("rtdetr-l.pt") # Vision Transformer Model
        # self.model = YOLO("yolov8l.pt") # Large YOLOv8 Model
        # self.model = YOLO("yolov8n.pt") # Nano YOLOv8 Model
        self.model = YOLO("/Users/jakobstrozberg/Documents/GitHub/SystemsEngineeringDroneDetection/best.pt")
        
        # Get the class names for the YOLO model
        self.CLASS_NAMES_DICT = ['Drone']
        
        print(self.CLASS_NAMES_DICT)
    
        # Initialize the box annotator for visualizing object detections
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

    
    def plot_bboxes(self, results, frame):
        
        # Extract detections from the YOLO model results
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        
        # Convert class IDs to integers
        class_id = class_id.astype(np.int32)
    
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=class_id,
                    )
    
        # Format custom labels for the object detections
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for xyxy, mask, confidence, class_id, track_id in detections]
        # Annotate and display the frame with the object detections
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    
    
    def __call__(self):
        # Initialize the screen capture device
        sct = mss()
        # Set the dimensions of the screen capture
        screen_dimensions = {"top": 135, "left": 0, "width": 810, "height": 800}
      
        #ser = Serial('/dev/cu.usbmodem2101, 115200', timeout  = 5) 
 ### ser = serial.Serial('/dev/cu.usbmodem2101', 9600, timeout=5) 
        
        while True:
            start_time = time()

            # Capture a frame from the screen
            screenshot = sct.grab(screen_dimensions)
            
            # Convert the screenshot to a numpy array and then to a color image
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Resize frame to match model's input size
            #frame = cv2.resize(frame, (512, 512))
            #print(frame.shape)


            # Use the YOLO model to predict object detections in the frame
            results = self.model.predict(frame, conf=0.5)
            
            # Annotate the frame with the object detections
            frame = self.plot_bboxes(results, frame)
  ###          if 'Drone' in self.labels:
  ###              ser.write("hello") 
 ###               print("Drone Detected!")
            
            # Calculate the frame rate and display it on the frame
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            # Display the annotated frame
            cv2.imshow('Drone Detection', frame)
 
            # Exit the loop if the user presses the ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        # Release the video capture device and destroy all windows
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index=1)
detector()