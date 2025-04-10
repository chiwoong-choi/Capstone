import os
import cv2
import mediapipe as mp
import time
import numpy as np
import csv
import threading

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

class CameraThread:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise ValueError(f"카메라 {camera_id}를 열 수 없습니다. 연결 상태를 확인하세요.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화

        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        
        # 스레드 실행
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("⚠️ [WARNING] 프레임을 읽는 데 실패했습니다.")
            time.sleep(0.01)  # CPU 점유율을 너무 높이지 않기 위해 잠시 대기

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# 모범 자세 프레임 저장
front_save_image_dir = "front_squat_frames"
right_save_image_dir = "right_squat_frames"
front_input_dir = "./front_squat_frames"
right_input_dir = "./right_squat_frames"
front_silhouette_dir = "./front_silhouette"
right_silhouette_dir = "./right_silhouette"
os.makedirs(front_save_image_dir, exist_ok=True)
os.makedirs(right_save_image_dir, exist_ok=True)

# 결과 프레임 저장 
result_save_image_dir = "frames"
front_result_output_video_path = "front_result.mp4"
right_result_output_video_path = "right_result.mp4"


def clear_folders(*folders):
    """
    지정된 폴더의 모든 파일을 삭제합니다.
    폴더 자체는 유지됩니다.
    
    Args:
        *folders: 삭제할 파일이 있는 폴더 경로들
    """
    for folder in folders:
        if os.path.exists(folder):
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 파일 또는 심볼릭 링크 삭제
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)  # 빈 디렉토리 삭제
                except Exception as e:
                    print(f"❌ [ERROR] 파일 삭제 실패: {file_path}, {e}")
            print(f"✅ [INFO] 폴더 정리 완료: {folder}")
        else:
            print(f"⚠️ [WARNING] 폴더가 존재하지 않습니다: {folder}")
        
        
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
def recording_pose(front_camera, right_camera):
    """5초 동안 프레임을 저장"""
    front_frame_count = 0
    right_frame_count = 0
    recording_start_time = time.time()

    while time.time() - recording_start_time < 5:  # 5초 동안 녹화
        front_frame = front_camera.get_frame()
        right_frame = right_camera.get_frame()
        
        if front_frame is None or right_frame is None:
            print("⚠️ [WARNING] 카메라에서 프레임을 가져올 수 없습니다. 계속 시도 중...")
            time.sleep(0.1)  # 잠시 대기 후 다시 시도
            continue

        front_frame = cv2.resize(front_frame, (640, 480))
        right_frame = cv2.resize(right_frame, (640, 480))
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

def mk_dictionary(csv_file="squat_analysis.csv"):
    data_dict = {}  # CSV -> 메모리 저장용
    with open(csv_file, mode="r") as file:
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
    
    return data_dict

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

def control_squat_posture(data_dict):
    
    st_failed_front_frame_num = [] # 자세가 잘못된 frame 저장용
    ht_failed_front_frame_num = []
    kd_failed_front_frame_num = []
    knee_angle_flag = True
    min_knee_angle = min(value[2] for value in data_dict.values())
    min_angle_frame = min(data_dict, key=lambda k: data_dict[k][2])
    print(f"frame-{min_angle_frame} has min_angle: {min_knee_angle:.2f}")
    
    max_shoulder_distance = max(value[4] for value in data_dict.values())
    print(f"{max_shoulder_distance}")
    half_shoulder_distance = 0.5*max_shoulder_distance
    print(f"{half_shoulder_distance}")

    # 스쿼트 각도 
    if min_knee_angle > 80:
        print("Not Deep Squat")
        knee_angle_flag = False
        
    # 기타 각도 연산 
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
    
    
    return (unique_st_failed_front_frame_num, 
            unique_ht_failed_front_frame_num, 
            unique_kd_failed_front_frame_num, 
            min_angle_frame
            )

def show_result(unique_st_failed_front_frame_num, unique_ht_failed_front_frame_num, unique_kd_failed_front_frame_num, min_angle_frame):
    front_frame_files = sorted(os.listdir(front_save_image_dir))  # 정렬하여 순서대로 재생
    right_frame_files = sorted(os.listdir(right_save_image_dir))
    
    total_frames = len(front_frame_files)  # 총 프레임 수
    total_duration = 10  # 총 재생 시간 (초)
    delay = int((total_duration / total_frames) * 1000)  # 각 프레임의 딜레이 (ms)

    front_first_frame = cv2.imread(os.path.join(front_save_image_dir, front_frame_files[0]))
    right_first_frame = cv2.imread(os.path.join(right_save_image_dir, right_frame_files[0]))
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
            print(f"⚠️ [WARNING] Frame {frame_file} is None. Skipping...")
            continue

        # 🔹 틀린 프레임 강조 및 랜드마크 표시
        rgb_front_frame = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
        front_results = pose.process(rgb_front_frame)

        if front_results.pose_landmarks:
            landmarks = front_results.pose_landmarks.landmark
            front_height, front_width, _ = front_frame.shape

            # 랜드마크 좌표 계산
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

            # 틀린 프레임 강조
            if frame_file in unique_st_failed_front_frame_num:
                cv2.line(front_frame, shoulder_left, shoulder_right, (0, 0, 255), 3)  # 어깨 빨간선
            if frame_file in unique_ht_failed_front_frame_num:
                cv2.line(front_frame, hip_left, hip_right, (0, 0, 255), 3)  # 골반 빨간선
            if frame_file in unique_kd_failed_front_frame_num:
                cv2.line(front_frame, knee_left, knee_right, (0, 0, 255), 3)  # 무릎 빨간선

        # 비디오 저장
        front_out.write(front_frame)
        right_out.write(right_frame)

        # 화면 출력
        try:
            cv2.imshow('Front Result', front_frame)
            cv2.imshow('Right Result', right_frame)
        except Exception as e:
            print(f"❌ [ERROR] Failed to display frame {frame_file}: {e}")
            continue

        # 모든 프레임에 동일한 딜레이 적용
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    front_out.release()
    right_out.release()
    cv2.destroyAllWindows()

def create_silhouettes(front_input_dir, right_input_dir, front_silhouette_dir, right_silhouette_dir):
    # MediaPipe Selfie Segmentation 모델 로드
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # 출력 디렉토리 생성
    if not os.path.exists(front_silhouette_dir):
        os.makedirs(front_silhouette_dir)

    if not os.path.exists(right_silhouette_dir):
        os.makedirs(right_silhouette_dir)

    # 이미지 파일들 정렬
    front_image_files = sorted(os.listdir(front_input_dir))  # front_squat_frames 디렉토리에서 이미지 파일 정렬
    right_image_files = sorted(os.listdir(right_input_dir))  # right_squat_frames 디렉토리에서 이미지 파일 정렬

    # 전면 실루엣 생성
    for img_name in front_image_files:
        img_path = os.path.join(front_input_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # RGB 변환 후 MediaPipe 실행
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segment.process(frame_rgb)

        # 마스크 생성 (사람 부분 1, 배경 0)
        mask = results.segmentation_mask
        _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 실루엣 생성
        silhouette_frame = np.zeros_like(frame)
        silhouette_frame[binary_mask > 0] = (0, 0, 255)  # 실루엣을 파란색으로 표시

        # 결과 저장
        output_path = os.path.join(front_silhouette_dir, img_name)
        cv2.imwrite(output_path, silhouette_frame)

    print(f"전면 실루엣 {len(front_image_files)}개를 저장했습니다.")

    # 우측면 실루엣 생성
    for img_name in right_image_files:
        img_path = os.path.join(right_input_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # RGB 변환 후 MediaPipe 실행
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segment.process(frame_rgb)

        # 마스크 생성 (사람 부분 1, 배경 0)
        mask = results.segmentation_mask
        _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 실루엣 생성
        silhouette_frame = np.zeros_like(frame)    
        silhouette_frame[binary_mask > 0] = (0, 0, 255)  # 실루엣을 빨간색으로 표시

        # 결과 저장
        output_path = os.path.join(right_silhouette_dir, img_name)
        cv2.imwrite(output_path, silhouette_frame)

    print(f"우측 실루엣 {len(right_image_files)}개를 저장했습니다.")
            

def realtime_synchronize(data_dict, front_camera_thread, right_camera_thread):
    print("🔵 [INFO] 실루엣 동기화 시작")

    # 실루엣 이미지 캐싱
    front_silhouettes = [cv2.imread(os.path.join(front_silhouette_dir, f)) for f in sorted(os.listdir(front_silhouette_dir))]
    right_silhouettes = [cv2.imread(os.path.join(right_silhouette_dir, f)) for f in sorted(os.listdir(right_silhouette_dir))]

    silhouette_angle = np.array([value[2] for value in data_dict.values()])
    squat_count = 0
    is_squatting = False
    frame_counter = 0
    process_interval = 5  # 5 프레임마다 처리
    previous_knee_angle = None  # 이전 프레임의 다리 각도
    previous_front_silhouette = None  # 이전에 사용한 전면 실루엣
    previous_right_silhouette = None  # 이전에 사용한 우측 실루엣
    angle_threshold = 1  # 다리 각도 변화 임계값 (5도 이하일 경우 동일한 실루엣 사용)

    while True:
        frame_front = front_camera_thread.get_frame()
        frame_right = right_camera_thread.get_frame()

        if frame_front is None or frame_right is None:
            print("⚠️ [WARNING] 실시간 프레임을 가져올 수 없습니다. 종료합니다.")
            break

        frame_front_rgb = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

        # MediaPipe 처리 간격 조정
        frame_counter += 1
        if frame_counter % process_interval == 0:
            results_front = pose.process(frame_front_rgb)
            results_right = pose.process(frame_right_rgb)
        else:
            results_front = None
            results_right = None

        if results_front and results_front.pose_landmarks:
            landmarks = results_front.pose_landmarks.landmark
            hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
            knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
            ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

            knee_angle = calculate_angle_knee(hip, knee, ankle)
            print(f"🔢 [INFO] 현재 다리 각도: {knee_angle:.2f}")

            # 스쿼트 카운팅 로직
            if knee_angle > 140:
                if is_squatting:
                    squat_count += 1
                    print(f"✅ [INFO] 스쿼트 카운트: {squat_count}")
                    is_squatting = False
            elif knee_angle < 100:
                is_squatting = True

            # 다리 각도 변화가 거의 없는 경우 이전 실루엣 사용
            if previous_knee_angle is not None and abs(knee_angle - previous_knee_angle) <= angle_threshold:
                front_silhouette = previous_front_silhouette
                right_silhouette = previous_right_silhouette
            else:
                # 실루엣 선택
                closest_frame_index = np.argmin(np.abs(silhouette_angle - knee_angle))
                front_silhouette = front_silhouettes[closest_frame_index]
                right_silhouette = right_silhouettes[closest_frame_index]

                # 이전 실루엣 업데이트
                previous_front_silhouette = front_silhouette
                previous_right_silhouette = right_silhouette

            # 이전 다리 각도 업데이트
            previous_knee_angle = knee_angle

            # NumPy 배열 연산으로 오버레이
            alpha = 0.7
            beta = 0.3
            front_overlay = (alpha * frame_front + beta * front_silhouette).astype(np.uint8)
            right_overlay = (alpha * frame_right + beta * right_silhouette).astype(np.uint8)

            # 화면 표시
            cv2.imshow('Front View (Overlay)', front_overlay)
            cv2.imshow('Right View (Overlay)', right_overlay)

        if squat_count >= 10:
            print("[INFO] 10회 이상 스쿼트 완료")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔴 [INFO] 사용자에 의해 종료되었습니다.")
            break

    cv2.destroyAllWindows()
    
# 실행 코드
if __name__ == "__main__":
    clear_folders(front_save_image_dir, right_save_image_dir, front_silhouette_dir, right_silhouette_dir)
    front_camera = CameraThread(0)
    right_camera = CameraThread(1)

    try:
        print("🔵 [INFO] 프로그램 시작")
        while True:
            #key = cv2.waitKey(1) & 0xFF
            #if key == ord('w'):  # Space bar 입력 감지
                print("🔵 [INFO] 초기 작업 시작")
                time.sleep(3)  # 3초 대기
                recording_pose(front_camera, right_camera)
                analyse_pose()
                data_dict = mk_dictionary()
                st_failed, ht_failed, kd_failed, min_angle_frame = control_squat_posture(data_dict)
                show_result(st_failed, ht_failed, kd_failed, min_angle_frame)

                create_silhouettes(
                    front_input_dir=front_input_dir,
                    right_input_dir=right_input_dir,
                    front_silhouette_dir=front_silhouette_dir,
                    right_silhouette_dir=right_silhouette_dir,
                )
                print("🔵 [INFO] 운동 시작")
                
                realtime_synchronize(data_dict, front_camera, right_camera)
                
                break
                
    except Exception as e:
        print(f"❌ [ERROR] 예외 발생: {e}")

    finally:
        front_camera.stop()
        right_camera.stop()
        cv2.destroyAllWindows()