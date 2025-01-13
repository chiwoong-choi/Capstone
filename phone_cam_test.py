import cv2
import mediapipe as mp
import time


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,    
    model_complexity=0,         
    smooth_landmarks=True,      
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5    
)


cap = cv2.VideoCapture("http://192.168.1.3:4747/video")  
if not cap.isOpened():
    print("shit")
    exit()


frame_count = 0
start_time = time.time()


frame_skip = 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("no")
        break

    frame_count += 1
   
    if frame_count % frame_skip != 0:
        continue

   
    frame = cv2.resize(frame, (320, 240))

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
    results = pose.process(rgb_frame)

    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    
    cv2.imshow('MediaPipe Pose', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

    

