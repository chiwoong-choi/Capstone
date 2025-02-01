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
    print("카메라 스트림을 열 수 없습니다.")
    raise Exception("카메라 스트림을 열 수 없습니다.")

# 변수 초기화
recording = False
recording_start_time = None
frames = []
save_video_path = "squat_pose_{}.mp4"  # 여러 비디오를 저장하기 위해 파일 이름에 인덱스 추가

frame_count = 0
start_time = time.time()
frame_skip = 2
video_count = 0  # 비디오 저장 횟수를 추적하는 카운터

def detect_squat_condition(landmarks):
    """
    if로 조건 추가하여 스쿼트 자세 확인. 현재는 조건없이 영상저장중
    """
    return True  # 스쿼트를 하는 조건이 맞다면 True 반환


def save_video(frames, video_count):
    
    #저장할 비디오를 생성하여 지정된 경로에 저장.
    
    save_path = save_video_path.format(video_count)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 20, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"비디오 저장 완료: {save_path}")

def main():
    global recording, recording_start_time, frames, video_count,frame_count
    
    while True:
        try:
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

                # 특정 조건을 만족하면 녹화 시작
                if detect_squat_condition(results.pose_landmarks.landmark):
                    if not recording:  # 스쿼트를 처음 시작할 때
                        recording = True
                        recording_start_time = time.time()
                        frames = []  # 프레임 초기화
                        print("스쿼트 시작, 녹화 중...")

                # 녹화 중일 때 프레임 저장
                if recording:
                    frames.append(frame)

                    # 5초 경과 시 녹화 종료
                    if time.time() - recording_start_time >= 5:
                        recording = False
                        video_count += 1
                        print("5초 영상 종료, 비디오 저장 중...")
                        save_video(frames, video_count)
                        frames = []  # 비디오가 저장된 후 프레임 초기화

            # 화면 출력
            cv2.imshow('MediaPipe Pose', frame)

            # 종료 조건 (q 키를 누르면 종료)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"오류 발생: {e}")
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

