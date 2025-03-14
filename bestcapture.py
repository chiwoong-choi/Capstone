import os
import cv2
import mediapipe as mp
import time
import numpy as np
import csv

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 카메라 연결
front_cap = cv2.VideoCapture("exam3.mp4")
right_cap = cv2.VideoCapture("exam4.mp4")
#front_cap = cv2.VideoCapture(0)
#right_cap = cv2.VideoCapture("http://192.168.81.137:4747/video")

if not front_cap.isOpened() or not right_cap.isOpened:
    raise Exception("카메라 스트림을 열 수 없습니다.")

# 모범 자세 프레임 저장
front_save_image_dir = "front_squat_frames"
right_save_image_dir = "right_squat_frames"
os.makedirs(front_save_image_dir, exist_ok=True)
os.makedirs(right_save_image_dir, exist_ok=True)

# 결과 프레임 저장 
result_save_image_dir = "frames"
front_result_output_video_path = "front_result.mp4"
right_result_output_video_path = "right_result.mp4"

# 다리 각도 계산 함수
def calculate_angle_knee(a, b, c):
    """세 점을 이용해 무릎 각도를 계산"""
    vector_ab = np.array([a[0] - b[0], a[1] - b[1]])  # 벡터 a -> b
    vector_bc = np.array([c[0] - b[0], c[1] - b[1]])  # 벡터 b -> c
    dot_product = np.dot(vector_ab, vector_bc)
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_bc = np.linalg.norm(vector_bc)
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

# 골반 기울기 계산 함수
def calculate_tilt_hip(hip_left, hip_right):
    hip_vect = np.array([hip_right[0] - hip_left[0], hip_right[1]-hip_left[1]])
    ground_vect = np.array([1, 0])  # 지면 기준 벡터
    dot_product_hip = np.dot(hip_vect, ground_vect)
    magnitude_hip_vect = np.linalg.norm(hip_vect)
    cos_hip = dot_product_hip / magnitude_hip_vect
    angle_hip = np.degrees(np.arccos(np.clip(cos_hip, -1.0, 1.0)))
    return angle_hip

# 어깨 기울기 계산 함수
def calculate_tilt_shoulder(shoulder_left, shoulder_right):
    """어깨의 기울기 각도를 계산"""
    shoulder_vect = np.array([shoulder_right[0] - shoulder_left[0], shoulder_right[1] - shoulder_left[1]])
    ground_vect = np.array([1, 0])  # 지면 기준 벡터
    dot_product_shoulder = np.dot(shoulder_vect, ground_vect)
    magnitude_shoulder_vect = np.linalg.norm(shoulder_vect)
    cos_shoulder = dot_product_shoulder / magnitude_shoulder_vect
    angle_shoulder = np.degrees(np.arccos(np.clip(cos_shoulder, -1.0, 1.0)))
    return angle_shoulder

# 무릎 거리 계산 함수
def calculate_knee_dis(knee_left, knee_right):
    dis_knee = np.abs(knee_left[0] - knee_right[0])
    return dis_knee

# 어깨 거리 계산 함수
def calculate_shoulder_dis(shoulder_left, shoulder_right):
    dis_shoulder= np.abs(shoulder_left[0] - shoulder_right[0])
    return dis_shoulder

# 프레임 저장 함수
def recording_pose():
    """5초 동안 프레임을 저장"""
    front_frame_count = 0
    right_frame_count = 0
    recording_start_time = time.time()

    while time.time() - recording_start_time < 5:  # 5초 동안 녹화
        front_ret, front_frame = front_cap.read()
        right_ret, right_frame = right_cap.read()
        
        if not front_ret or not right_ret:
            print("카메라에서 프레임을 가져올 수 없습니다.")
            break

        front_frame = cv2.resize(front_frame, (640, 480))
        right_frame = cv2.resize(right_frame, (640, 480))
        #frame_filename = os.path.join(save_image_dir, f"frame_{frame_count:04d}.png")
        front_frame_filename = os.path.join(front_save_image_dir, f"{front_frame_count:04d}.png")
        right_frame_filename = os.path.join(right_save_image_dir, f"{right_frame_count:04d}.png")
        cv2.imwrite(front_frame_filename, front_frame)  # 이미지 저장
        cv2.imwrite(right_frame_filename, right_frame)
        front_frame_count += 1
        right_frame_count += 1

        cv2.imshow('Camera 1 - Front Squat', front_frame)
        cv2.imshow('Camera 2 - Right Squat', right_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("녹화 완료. 프레임 저장 완료.")

# 랜드마크 분석 함수
def analyse_pose():
    """저장된 프레임을 불러와 무릎 각도와 어깨 기울기를 분석하고 결과를 즉시 저장"""
    # CSV 파일로 결과 저장
    with open("squat_analysis.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame File", "Knee Distance", "Shoulder Tilt", "Knee Angle","Hip Tilt"])  # CSV 헤더

        front_frame_files = sorted(os.listdir(front_save_image_dir))  # 정렬하여 순서대로 분석

        for frame_file in front_frame_files:
            frame_path = os.path.join(front_save_image_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 좌표 추출
                hip_left = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                hip_right = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                knee_left = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
                knee_right = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
                ankle_left = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)
                shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

                # 각도 계산
                knee_angle = calculate_angle_knee(hip_left, knee_left, ankle_left)
                shoulder_tilt = calculate_tilt_shoulder(shoulder_left, shoulder_right)
                knee_distance = calculate_knee_dis(knee_left, knee_right)
                hip_tilt = calculate_tilt_hip(hip_left, hip_right)
                shoulder_distance = calculate_shoulder_dis(shoulder_right, shoulder_left)
                # 분석 결과를 CSV에 저장
                writer.writerow([frame_file, knee_distance, shoulder_tilt, knee_angle, hip_tilt,shoulder_distance])
                print(f"{frame_file}: Knee Distance = {knee_distance:.2f}, Shoulder Tilt = {shoulder_tilt:.2f}, Knee_angle = {knee_angle:.2f}, Hip_Tilt = {hip_tilt:.2f}, Shoulder_Distance = {shoulder_distance:.2f}")
                

    print("각도 분석 완료. 결과 저장됨: squat_analysis.csv")

def control_squat_posture():
    
    data_dict = {} #csv -> memory 저장용
    st_failed_front_frame_num = [] # 자세가 잘못된 frame 저장용
    ht_failed_front_frame_num = []
    kd_failed_front_frame_num = []
    
    # CSV 파일에서 결과 읽기
    with open("squat_analysis.csv", mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 건너뛰기
        
        for row in reader:
            frame_file, knee_dist, shoulder_tilt, knee_angle, hip_tilt, shoulder_distance = row
            knee_dist = float(knee_dist)
            shoulder_tilt = float(shoulder_tilt)
            knee_angle = float(knee_angle)
            hip_tilt = float(hip_tilt)
            shoulder_distance = float(shoulder_distance)
            data_dict[frame_file] = (knee_dist, shoulder_tilt, knee_angle, hip_tilt, shoulder_distance)
    
    min_knee_angle = min(value[2] for value in data_dict.values())
    min_angle_frame = min(data_dict, key=lambda k: data_dict[k][2])
    print(f"frame-{min_angle_frame} has min_angle: {min_knee_angle:.2f}")
    
    
    max_shoulder_distance = max(value[4] for value in data_dict.values())
    print(f"{max_shoulder_distance}")
    half_shoulder_distance = 0.5*max_shoulder_distance
    print(f"{half_shoulder_distance}")

    #다리 각도 -> 스쿼트 여부 판별
    if min_knee_angle > 80:
        print("Not Deep Squat")
        
        
    for frame_file, (knee_dist, shoulder_tilt, knee_angle, hip_tilt,shoulder_distance) in data_dict.items():
        #어깨 불균형을 바로 고지함
        if shoulder_tilt < 177:
            print("Shoulder Tilted")
            st_failed_front_frame_num.append(frame_file) #자세가 잘못된 프레임 번호 저장
            
        if hip_tilt < 177:
            print("Hip Tilted")
            ht_failed_front_frame_num.append(frame_file) #자세가 잘못된 프레임 번호 저장
        
        if knee_dist > 1.2*max_shoulder_distance:
            print("Knee So Far")
            kd_failed_front_frame_num.append(frame_file) #자세가 잘못된 프레임 번호 저장  
            
        if knee_dist < half_shoulder_distance:
            print("Knee So Close")
            kd_failed_front_frame_num.append(frame_file) #자세가 잘못된 프레임 번호 저장
            
    #failed_front_frame_num = list(set(failed_front_frame_num))        
    
    unique_st_failed_front_frame_num = list(dict.fromkeys(st_failed_front_frame_num))
    unique_ht_failed_front_frame_num = list(dict.fromkeys(ht_failed_front_frame_num))
    unique_kd_failed_front_frame_num = list(dict.fromkeys(kd_failed_front_frame_num))
    print(f"{unique_st_failed_front_frame_num}") #리스트 확인용
    print(f"{unique_ht_failed_front_frame_num}") #리스트 확인용
    print(f"{unique_kd_failed_front_frame_num}") #리스트 확인용
    
    
    return unique_st_failed_front_frame_num, unique_ht_failed_front_frame_num, unique_kd_failed_front_frame_num, min_angle_frame

def show_result(unique_st_failed_front_frame_num, unique_ht_failed_front_frame_num, unique_kd_failed_front_frame_num, min_angle_frame):
    front_frame_files = sorted(os.listdir(front_save_image_dir))  # 정렬하여 순서대로 재생
    right_frame_files = sorted(os.listdir(right_save_image_dir))
    normal_delay = 50  # 일반 프레임 재생 속도 (50ms)
    slow_delay = 450   # 문제 있는 프레임의 재생 속도 (450ms)

    front_first_frame = cv2.imread(os.path.join(front_save_image_dir, front_frame_files[0]))
    right_first_frame = cv2.imread(os.path.join(right_save_image_dir,right_frame_files[0]))
    front_height, front_width, _ = front_first_frame.shape
    right_height, right_width, _ = right_first_frame.shape
    front_fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱 설정
    right_fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱 설정

    front_out = cv2.VideoWriter(front_result_output_video_path, front_fourcc, 20.0, (front_width, front_height))
    right_out = cv2.VideoWriter(right_result_output_video_path, right_fourcc, 20.0, (right_width, right_height))

    for frame_file in front_frame_files:
        front_frame_path = os.path.join(front_save_image_dir, frame_file)
        right_frame_path = os.path.join(right_save_image_dir, frame_file)
        front_frame = cv2.imread(front_frame_path)
        right_frame = cv2.imread(right_frame_path)
        if front_frame is None or right_frame is None:
            continue
        
        rgb_front_frame = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
        rgb_right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        front_results = pose.process(rgb_front_frame)
        right_results = pose.process(rgb_right_frame)
        # 프레임이 문제 있는 경우 강조 표시
        if frame_file in unique_st_failed_front_frame_num or \
           frame_file in unique_ht_failed_front_frame_num or \
           frame_file in unique_kd_failed_front_frame_num or \
               frame_file == min_angle_frame:
            
            if front_results.pose_landmarks:
                landmarks = front_results.pose_landmarks.landmark
                front_height, front_width, _ = front_frame.shape

                # 어깨, 골반, 무릎 좌표 추출
                shoulder_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * front_width),
                                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * front_height))
                shoulder_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * front_width),
                                  int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * front_height))
                hip_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * front_width),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * front_height))
                hip_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * front_width),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * front_height))
                knee_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * front_width),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * front_height))
                knee_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * front_width),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * front_height))

                # 문제 유형별 강조 (빨간 선 추가)
                if frame_file in unique_st_failed_front_frame_num:
                    cv2.line(front_frame, shoulder_left, shoulder_right, (0, 0, 255), 3)  # 어깨 빨간선
                if frame_file in unique_ht_failed_front_frame_num:
                    cv2.line(front_frame, hip_left, hip_right, (0, 0, 255), 3)  # 골반 빨간선
                if frame_file in unique_kd_failed_front_frame_num:
                    cv2.line(front_frame, knee_left, knee_right, (0, 0, 255), 3)  # 무릎 빨간선

            if right_results.pose_landmarks:
                landmarks_right = right_results.pose_landmarks.landmark
                right_height, right_width, _ = right_frame.shape

                # 어깨, 골반, 무릎 좌표 추출 (right frame)            
                hip_right = (int(landmarks_right[mp_pose.PoseLandmark.RIGHT_HIP].x * right_width),
                             int(landmarks_right[mp_pose.PoseLandmark.RIGHT_HIP].y * right_height))
                knee_left = (int(landmarks_right[mp_pose.PoseLandmark.LEFT_KNEE].x * right_width),
                             int(landmarks_right[mp_pose.PoseLandmark.LEFT_KNEE].y * right_height))
                knee_right = (int(landmarks_right[mp_pose.PoseLandmark.RIGHT_KNEE].x * right_width),
                              int(landmarks_right[mp_pose.PoseLandmark.RIGHT_KNEE].y * right_height))
                ankle_right = (int(landmarks_right[mp_pose.PoseLandmark.RIGHT_ANKLE].x* right_width),
                              int(landmarks_right[mp_pose.PoseLandmark.RIGHT_ANKLE].y* right_height))
                if frame_file in unique_st_failed_front_frame_num:
                    cv2.line(right_frame, hip_right, knee_right, (0, 0, 255), 3)  # 다리 각도 빨간선
                    cv2.line(right_frame, knee_right, ankle_right, (0, 0, 255), 3)  # 다리 각도 빨간선
                    
        # 비디오 저장 (속도에 맞춰 저장)
        if frame_file in unique_st_failed_front_frame_num or \
           frame_file in unique_ht_failed_front_frame_num or \
           frame_file in unique_kd_failed_front_frame_num:
            # 문제 있는 프레임은 여러 번 반복해서 저장 (속도 느리게)
            for _ in range(9):  # 프레임을 9번 반복 저장하여 느리게 표시 (1번 당 450ms)
                front_out.write(front_frame)
                right_out.write(right_frame)
        else:
            # 일반 프레임은 1번만 저장
            front_out.write(front_frame)
            right_out.write(right_frame)
        # 화면 출력
        cv2.imshow('Front View', front_frame)
        cv2.imshow('Right View', right_frame)

        # 딜레이 설정 (잘못된 프레임이면 slow_delay 적용)
        delay = slow_delay if frame_file in unique_st_failed_front_frame_num or \
                            frame_file in unique_ht_failed_front_frame_num or \
                            frame_file in unique_kd_failed_front_frame_num else normal_delay
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

#def silhouette():
    

# 실행 코드
if __name__ == "__main__":
    recording_pose()         # 1️⃣ 프레임 저장
    analyse_pose()           # 2️⃣ 프레임 분석 및 즉시 저장
    st_failed, ht_failed, kd_failed, min_angle_frame =  control_squat_posture()  # 3️⃣ 최적 자세 찾기
    show_result(st_failed, ht_failed, kd_failed, min_angle_frame)
    front_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()
