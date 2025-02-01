import os
import cv2
import mediapipe as mp
import time

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 카메라 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("카메라 스트림을 열 수 없습니다.")

recording = False
recording_start_time = None
frames = []
save_video_path = "squat_pose.mp4"

frame_count = 0
start_time = time.time()
frame_skip = 2


while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 가져올 수 없습니다.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))  # 프레임 크기 조정
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if not recording:  
            recording = True
            recording_start_time = time.time()
            frames = []  # 프레임 초기화
            print("녹화 시작")

        # 녹화 중일 때 프레임 저장
        if recording:
            frames.append(frame)

            # 녹화 시간이 5초 이상이면 저장
            if recording_start_time is not None and time.time() - recording_start_time >= 5:
                recording = False
                print("비디오 저장 중")

                # 비디오 저장
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_video_path, fourcc, 20, (width, height))

                for f in frames:
                    out.write(f)
                out.release()
                print(f"비디오 저장 완료: {save_video_path}")



    # 화면 출력
    cv2.imshow('MediaPipe Pose', frame)

    # 종료 조건 (q 키를 누르면 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
