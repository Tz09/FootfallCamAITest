import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Initialize
model = YOLO('best.pt')     # Load custom model
model.to('cuda')            # Use GPU 
model.names['2'] = 'staff'  # Add Staff to the class names

def detect(video_path,conf=0.25):
    # Initialize video file and confidence score
    video_path = video_path
    conf=conf
    
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print('Error opening the video file.')

    while cap.isOpened():
        success, frame = cap.read()
        key = cv2.waitKey(1) & 0xFF
        if success:

            # Run YOLOv8 inference on the frame
            results = model.predict(frame,conf=conf,verbose=False)

            # Change to Numpy for easier manipulation
            boxes = results[0].boxes
            boxes = boxes.to('cpu')
            boxes = boxes.numpy()

            # Get the Label
            label = [box for box in boxes if box.cls == 0]

            # Record the coordinates of the staff
            staff_coordinate = []

            # Loop through each Label
            # If Label is inside the bounding box of the staff, update the class of the corresponding bounding box to 2.0 (staff)
            for label_box in label:
                for box in boxes:
                    if box.cls == 1.0:
                        if (label_box.xyxy[0][0] >= box.xyxy[0][0] and
                            label_box.xyxy[0][1] >= box.xyxy[0][1] and
                            label_box.xyxy[0][2] <= box.xyxy[0][2] and
                            label_box.xyxy[0][3] <= box.xyxy[0][3]):
                            box.boxes[:, -1] = 2.0
                            staff_coordinate.append(box.xyxy[0].tolist())

            # Update the whole result with appended boxes
            results[0].update(boxes.boxes)
            
            if len(staff_coordinate) != 0:
                print('--------------------------------------------------------------------------------------------------')
                for i, coordinate in enumerate(staff_coordinate, start=1):
                    print(f"Staff {i} Coordinate: {coordinate}")
                print('--------------------------------------------------------------------------------------------------\n')

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("Staff Identification", annotated_frame)

            # Break the loop if 'q' is pressed, Pause the video if 'p' is pressed
            if key == ord("q"):
                break
            elif key == ord("p"):
                cv2.waitKey(-1) #wait until any key is pressed
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Fill the video path and confidence score
    detect('sample.mp4',conf=0.5)