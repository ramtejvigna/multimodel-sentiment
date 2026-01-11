import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from mtcnn import MTCNN
import cv2
import numpy as np

detector = MTCNN(device='cpu')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_small = cv2.resize(frame_rgb, (640, 480))  # Keep original size
    
    # SAFE DETECTION - Check for empty results
    try:
        faces = detector.detect_faces(frame_small, confidence=0.5)
        
        # Skip if no faces detected (prevents crash)
        if len(faces) == 0:
            cv2.imshow('Real-time Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
    except Exception as e:
        print(f"Detection failed: {e}")
        cv2.imshow('Real-time Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Scale and draw (only if faces found)
    scale_x = frame.shape[1] / frame_small.shape[1]
    scale_y = frame.shape[0] / frame_small.shape[0]
    
    for face in faces:
        x, y, w, h = face['box']
        x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        confidence = face['confidence']
        cv2.putText(frame, f'{confidence:.2f}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Real-time Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()