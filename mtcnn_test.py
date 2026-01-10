from mtcnn import MTCNN
import cv2
import numpy as np

detector = MTCNN()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    
    # Skip failed frames
    if not ret or frame is None:
        continue
    
    # Resize for speed (MTCNN works fine on smaller frames)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_small = cv2.resize(frame_rgb, (640, 480))
    
    # Safe detection with confidence threshold
    faces = detector.detect_faces(frame_small, confidence=0.5)
    
    # Draw rectangles on original frame (scale coordinates)
    scale_x = frame.shape[1] / frame_small.shape[1]
    scale_y = frame.shape[0] / frame_small.shape[0]
    
    for face in faces:
        x, y, w, h = face['box']
        x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Confidence score
        confidence = face['confidence']
        cv2.putText(frame, f'{confidence:.2f}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Real-time Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()