import cv2
import numpy as np
import time

def get_rotation_angle(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    
   
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=10)
    display_frame = frame.copy()
    
    if lines is None:
        return 0, display_frame 
    
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
      
        if abs(angle) < 90:  
            angles.append(angle)
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 감지된 선을 초록색으로 표시
    
    if len(angles) == 0:
        return 0, display_frame 
    
    median_angle = np.median(angles)
    return median_angle, display_frame

def correct_camera_tilt(frame, angle):
   
   
    if abs(angle) < 0.1: 
        return frame
    
    h, w = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    corrected_frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
    return corrected_frame

if __name__ == "__main__":
    video_path = "squat_pose_2.mp4"  
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    if fps == 0:
        fps = 30  
    frame_delay = int(1000 / fps) 
    
    last_check_time = time.time()
    rotation_angle = 0  
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        if current_time - last_check_time >= 0.3: 
            rotation_angle, debug_frame = get_rotation_angle(frame)
            last_check_time = current_time
        else:
            debug_frame = frame.copy()
        
        corrected_frame = correct_camera_tilt(frame, rotation_angle)
        
        cv2.imshow("Detected Lines", debug_frame)  
        cv2.imshow("Corrected Video", corrected_frame)
        
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
