import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 빈 dataframe 생성
df = pd.DataFrame()

# webcam / video 불러오기
cap = cv2.VideoCapture(r'C:\capstone\squat_pose_4.mp4')
 # 경로 수정

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  
  while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      break  # 비디오가 끝나면 종료
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # pose data 저장
    landmark_indices = [0, 12, 14, 16, 22, 24, 26, 28, 30, 32]
    x = []
    
    # k = landmarks 개수 및 번호
    for k in landmark_indices:
        x.append(results.pose_landmarks.landmark[k].x)
        x.append(results.pose_landmarks.landmark[k].y)
        x.append(results.pose_landmarks.landmark[k].z)
        

    # list x를 dataframe으로 변경
    tmp = pd.DataFrame(x).T

    # dataframe에 정보 쌓아주기
    df = pd.concat([df, tmp])
    df.to_csv('pose_landmarks.csv', index=False)
    # 이미지가 성공적으로 로드되었을 때만 표시
    if success:
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        
        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()  # 창을 닫을 때 꼭 호출
