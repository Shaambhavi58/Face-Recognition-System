import cv2
import numpy as np
import dlib

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = cap.read()
    
    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    i = 0
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        
        i = i + 1
        
        # Display the face number on the screen
        cv2.putText(frame, 'Face num: ' + str(i), (x - 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        print("Face detected: ", i)
    
    # Show the frame with the face(s) detection
    cv2.imshow('frame', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
