# 수정 필요 사항: show_result 화면 비율 맞추기
# 추가사항: 혹시모를 정확도 플로팅
# 그 외 따로 문제 없는듯? 행복해~~

import cv2
import os
import time
import csv
import numpy as np  
from flask import Flask, render_template, send_from_directory, Response, request, redirect, url_for, abort
from threading import Thread, Lock
import mediapipe as mp
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor
from flask import session
from flask import jsonify
executor = ThreadPoolExecutor(max_workers=4)

#ffmpeg 인코딩버전

# ------------------- -------------------
front_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
right_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
front_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
front_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not front_camera.isOpened() or not right_camera.isOpened():
    print("⚠️ [ERROR] 카메라를 열 수 없습니다.")
    exit()


# ------------------- MediaPipe 초기화 -------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ------------------- 경로 변수 선언 -------------------
front_save_image_dir = "front_squat_frames"
right_save_image_dir = "right_squat_frames"
front_input_dir = "./front_squat_frames"
right_input_dir = "./right_squat_frames"
front_silhouette_dir = "./front_silhouette"
right_silhouette_dir = "./right_silhouette"
data_dict = {} 
current_set_status = {}
squat_count_status = {"count": 0, "set": 1}
os.makedirs(front_save_image_dir, exist_ok=True)
os.makedirs(right_save_image_dir, exist_ok=True)
total_frame_count = 0
mismatch_frame_count = 0
squat_count = 0 
is_squatting = False
warned_low_depth = False
not_deep_squat = 0
result_save_image_dir = "frames"
front_result_output_video_path = "front_result.mp4"
right_result_output_video_path = "right_result.mp4"
knee_angle_history = []
streaming_flags = {0: False, 1: False}
shared_silhouette_idx = None  # ✅ front → right 로 전달될 실루엣 인덱스
last_squat_time = 0
ANGLE_CHANGE_THRESHOLD = 5
direction = "up"
squat_started = False
accuracy_result = {
    "st_accuracy": None,
    "ht_accuracy": None,
    "kd_accuracy": None
}
# ------------------- 포즈 분석 함수 -------------------

#시연용 폴더 삭제 함수
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
        
# ------------------- 모범자세 저장 자료 생성  -------------------

def recording_pose(front_camera, right_camera):
    """5초 동안 프레임을 저장"""
    front_frame_count = 0
    right_frame_count = 0
    recording_start_time = time.time()
    frame_index = 0  # 전체 프레임 인덱스
    
    while time.time() - recording_start_time < 5:  # 5초 동안 녹화
        ret_front, front_frame = front_camera.read()
        ret_right, right_frame = right_camera.read()
        if not ret_front or not ret_right:
            print("⚠️ [WARNING] 카메라에서 프레임을 가져올 수 없습니다. 계속 시도 중...")
            time.sleep(0.1)
            continue
        
        if front_frame is None or right_frame is None:
            print("⚠️ [WARNING] 카메라에서 프레임을 가져올 수 없습니다. 계속 시도 중...")
            time.sleep(0.1)  # 잠시 대기 후 다시 시도
            continue
        if frame_index % 2 == 0:
            front_frame_filename = os.path.join(front_save_image_dir, f"{front_frame_count:04d}.png")
            right_frame_filename = os.path.join(right_save_image_dir, f"{right_frame_count:04d}.png")
            cv2.imwrite(front_frame_filename, front_frame)  # 이미지 저장
            cv2.imwrite(right_frame_filename, right_frame)
            front_frame_count += 1
            right_frame_count += 1
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("녹화 완료. 프레임 저장 완료.")


def mk_dictionary(csv_file="squat_analysis.csv"):
    global data_dict
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

# ------------------- 모범자세 분석 -------------------

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
    not_deep_squat = False
    st_failed_front_frame_num = [] # 자세가 잘못된 frame 저장용
    ht_failed_front_frame_num = []
    kd_failed_front_frame_num = []
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
        not_deep_squat = True
        
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
    global accuracy_result
    front_frame_files = sorted(os.listdir(front_save_image_dir))  # 정렬하여 순서대로 재생
    right_frame_files = sorted(os.listdir(right_save_image_dir))
    
    st_length = len(unique_st_failed_front_frame_num)
    ht_length = len(unique_ht_failed_front_frame_num)
    kd_length = len(unique_kd_failed_front_frame_num)
    
    total_frames = len(front_frame_files)  # 총 프레임 수
    
    st_accuracy = (total_frames - st_length) / total_frames * 100
    ht_accuracy = (total_frames - ht_length) / total_frames * 100
    kd_accuracy = (total_frames - kd_length) / total_frames * 100
    
    accuracy_result["st_accuracy"] = st_accuracy
    accuracy_result["ht_accuracy"] = ht_accuracy
    accuracy_result["kd_accuracy"] = kd_accuracy
    # 결과 저장 디렉토리
    result_save_dir = "static"
    os.makedirs(result_save_dir, exist_ok=True)
    
    # 임시 디렉토리 생성 (FFmpeg 입력용)
    temp_front_dir = os.path.join(result_save_dir, "temp_front")
    temp_right_dir = os.path.join(result_save_dir, "temp_right")
    os.makedirs(temp_front_dir, exist_ok=True)
    os.makedirs(temp_right_dir, exist_ok=True)

    # 프레임 처리 및 저장
    for i, frame_file in enumerate(front_frame_files):
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
        rgb_right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        right_results = pose.process(rgb_right_frame)
        
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
        # 임시 디렉토리에 프레임 저장
        front_temp_path = os.path.join(temp_front_dir, f"{i:04d}.png")
        right_temp_path = os.path.join(temp_right_dir, f"{i:04d}.png")
        cv2.imwrite(front_temp_path, front_frame)
        cv2.imwrite(right_temp_path, right_frame)

    # FFmpeg 명령어로 MP4 파일 생성
    front_result_output_video_path = os.path.join(result_save_dir, "front_result.mp4")
    right_result_output_video_path = os.path.join(result_save_dir, "right_result.mp4")

    ffmpeg_front_command = [
        "ffmpeg", "-y", "-framerate", "20", "-i", os.path.join(temp_front_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", front_result_output_video_path
    ]
    ffmpeg_right_command = [
        "ffmpeg", "-y", "-framerate", "20", "-i", os.path.join(temp_right_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", right_result_output_video_path
    ]

    # FFmpeg 실행
    subprocess.run(ffmpeg_front_command, check=True)
    subprocess.run(ffmpeg_right_command, check=True)

    # 임시 디렉토리 삭제
    for temp_dir in [temp_front_dir, temp_right_dir]:
        for file_name in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file_name))
        os.rmdir(temp_dir)

    print(f"✅ [INFO] 결과 동영상 저장 완료: {front_result_output_video_path}, {right_result_output_video_path}")
    print(f"정확도 확인용 로그:{st_accuracy},{ht_accuracy}, {kd_accuracy}")
    return st_accuracy, ht_accuracy, kd_accuracy

# ------------------- 모범자세 실루엣 생성  -------------------
def save_part_contour(points, height, width, part_path):
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        np.save(part_path, largest.squeeze(axis=1))  # (N,2) 형태
        
def generate_silhouette(input_dir, output_dir, segment, pose, body_parts, colors):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(os.listdir(input_dir))

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_segmentation = segment.process(frame_rgb)
        results_pose = pose.process(frame_rgb)

        if not results_pose.pose_landmarks:
            print(f"⚠️ [WARNING] 랜드마크를 찾을 수 없습니다: {img_name}")
            silhouette_frame = np.zeros_like(frame)
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, silhouette_frame)
            continue

        landmarks = results_pose.pose_landmarks.landmark
        height, width, _ = frame.shape
        landmark_points = {
            idx: (int(landmark.x * width), int(landmark.y * height))
            for idx, landmark in enumerate(landmarks) if landmark.visibility > 0.5
        }

        mask = results_segmentation.segmentation_mask
        _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##
        silhouette_frame = np.zeros_like(frame)
        cv2.drawContours(silhouette_frame, contours, -1, (255, 0, 0), thickness=5)  # 빨간색 외곽선
        
        assignments = {part: [] for part in body_parts}

        for contour in contours:
            if contour.ndim == 3 and contour.shape[2] == 2:
                contour = contour.squeeze(axis=1)
            elif contour.ndim != 2 or contour.shape[1] != 2:
                print(f"⚠️ [WARNING] 잘못된 컨투어 형식: {contour.shape}")
                continue

            for pt in contour:
                min_dist = float('inf')
                closest_part = None
                for part, indices in body_parts.items():
                    for idx in indices:
                        if idx in landmark_points:
                            dist = np.linalg.norm(np.array(pt) - np.array(landmark_points[idx]))
                            if dist < min_dist:
                                min_dist = dist
                                closest_part = part
                if closest_part:
                    assignments[closest_part].append(tuple(pt))

        for part, points in assignments.items():
            for pt in points:
                cv2.circle(silhouette_frame, pt, 3, colors[part], -1)
        
        base_name = os.path.splitext(img_name)[0]
        for part, points in assignments.items():
            if not points:
                continue
            part_path = os.path.join(output_dir, f"{base_name}_{part}.npy")
            save_part_contour(points, height, width, part_path)
            np.save(part_path, np.array(points))
            
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, silhouette_frame)
        
    print(f"✅ [INFO] 실루엣 {len(image_files)}개를 저장했습니다.")

def create_silhouettes(front_input_dir, right_input_dir, front_silhouette_dir, right_silhouette_dir):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    body_parts = {
        "head": [0],
        "left_arm": [11, 13, 15],
        "right_arm": [12, 14, 16],
        "left_leg": [23, 25, 27],
        "right_leg": [24, 26, 28]
        #"torso": [11, 12, 23, 24]
    }

    colors = {
        "head": (255, 0, 0),
        "left_arm": (255, 0, 0),
        "right_arm": (255, 0, 0),
        "left_leg": (255, 0, 0),
        "right_leg": (255, 0, 0)
        #"torso": (255, 0, 0)
    }

    # Generate front and right silhouettes
    generate_silhouette(front_input_dir, front_silhouette_dir, segment, pose, body_parts, colors)
    generate_silhouette(right_input_dir, right_silhouette_dir, segment, pose, body_parts, colors)


# ------------------- 실시간 실루엣 오버레이  -------------------

def cache_silhouette_npy_with_angle_and_direction(silhouette_dir, body_parts, data_dict):
    cache = {}
    frame_indices = set()
    for fname in os.listdir(silhouette_dir):
        if fname.endswith('.npy'):
            idx = fname.split('_')[0]
            frame_indices.add(idx)
    for idx in frame_indices:
        cache[idx] = {}
        for part in body_parts:
            npy_path = os.path.join(silhouette_dir, f"{idx}_{part}.npy")
            if os.path.exists(npy_path):
                cache[idx][part] = np.load(npy_path)
            else:
                cache[idx][part] = np.array([])
        # 추가: 각도/방향 정보 저장
        # data_dict의 key는 "0000.png" 형태일 수 있으니 idx+".png"로 접근
        frame_file = f"{idx}.png"
        if frame_file in data_dict:
            knee_angle = data_dict[frame_file][2]
            # 동작 방향 추정 (이전 프레임과 비교)
            prev_idx = f"{int(idx)-1:04d}"
            if f"{prev_idx}.png" in data_dict:
                prev_angle = data_dict[f"{prev_idx}.png"][2]
                direction = "down" if knee_angle < prev_angle else "up"
            else:
                direction = "down"
            cache[idx]["knee_angle"] = knee_angle
            cache[idx]["direction"] = direction
    return cache

# 서버 시작 시 캐싱
#data_dict = mk_dictionary()
#front_silhouette_cache = cache_silhouette_npy_with_angle_and_direction(front_silhouette_dir, ["head","left_arm","right_arm","left_leg","right_leg"], data_dict)
#right_silhouette_cache = cache_silhouette_npy_with_angle_and_direction(right_silhouette_dir, ["head","left_arm","right_arm","left_leg","right_leg"], data_dict)

def realtime_synchronize(camera_type, data_dict, front_camera, right_camera, reps=1, sets=3, current_set=1):
    print("🔵 [INFO] 실루엣 동기화 시작")
    global squat_count, not_deep_squat, shared_silhouette_idx
    #squat_count = 0
    silhouette_angle_list = np.array([value[2] for value in data_dict.values()])
    silhouette_keys = list(data_dict.keys())

    colors = {
        "head": (255, 0, 0),
        "left_arm": (255, 0, 0),
        "right_arm": (255, 0, 0),
        "left_leg": (255, 0, 0),
        "right_leg": (255, 0, 0)
    }
    body_parts = {
        "head": [0],
        "left_arm": [11, 13, 15],
        "right_arm": [12, 14, 16],
        "left_leg": [23, 25, 27],
        "right_leg": [24, 26, 28]
    }

    mp_pose = mp.solutions.pose
    pose_front = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                              min_detection_confidence=0.3, min_tracking_confidence=0.3)
    pose_right = mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                              min_detection_confidence=0.3, min_tracking_confidence=0.3)

    previous_knee_angle = None
    previous_silhouette_idx = None
    direction = "down"
    frame_counter = 0  

    while True:
        try:
            ret_front, front_frame = front_camera.read()
            ret_right, right_frame = right_camera.read()
            if not ret_front or front_frame is None or not ret_right or right_frame is None:
                time.sleep(0.01)
                continue
            
            frame_counter += 1
            if frame_counter % 2 != 0:
                continue

            # FRONT 처리 및 실루엣 인덱스 결정
            def process_front():
                nonlocal previous_knee_angle, previous_silhouette_idx, direction
                global shared_silhouette_idx
                frame_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
                results = pose_front.process(frame_rgb)
                if not results.pose_landmarks:
                    return front_frame, previous_knee_angle, shared_silhouette_idx

                landmarks = results.pose_landmarks.landmark
                width, height = front_frame.shape[1], front_frame.shape[0]
                hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height)
                knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * height)
                ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height)
                knee_angle = calculate_angle_knee(hip, knee, ankle)

                if previous_knee_angle is None or abs(knee_angle - previous_knee_angle) > ANGLE_CHANGE_THRESHOLD:
                    idx = np.argmin(np.abs(silhouette_angle_list - knee_angle))
                    silhouette_idx = os.path.splitext(silhouette_keys[idx])[0]

                    previous_knee_angle = knee_angle
                    previous_silhouette_idx = silhouette_idx
                    shared_silhouette_idx = silhouette_idx
                else:
                    silhouette_idx = previous_silhouette_idx
                
                silhouette = front_silhouette_cache.get(silhouette_idx, {})
                overlay = draw_silhouette_overlay(front_frame, silhouette, body_parts, colors, results)

                # 디버그 표시
                cv2.putText(overlay, f"Silhouette: {silhouette_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

                # 스쿼트 카운팅
                update_squat_count(knee_angle)
                return overlay, knee_angle, silhouette_idx

            def process_right():
                silhouette = right_silhouette_cache.get(shared_silhouette_idx, {})
                frame_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
                results = pose_right.process(frame_rgb)
                overlay = draw_silhouette_overlay(right_frame, silhouette, body_parts, colors, results)
                cv2.putText(overlay, f"Silhouette: {shared_silhouette_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                return overlay

            # 병렬 처리 실행
            future_front = executor.submit(process_front)
            future_right = executor.submit(process_right)
            processed_front, _, _ = future_front.result()
            processed_right = future_right.result()

            # 종료 조건
            if squat_count >= reps:
                
                total = squat_count + not_deep_squat
                accuracy = squat_count / total * 100 if total > 0 else 0
                print(f"✅ [INFO] 세트 완료 - 정확도: {accuracy:.1f}%")
                #squat_count = 0
                not_deep_squat = 0
                time.sleep(3) # 3초 대기 후 breaktime.html로 이동
                break

            # 결과 송출
            frame = processed_front if camera_type == "front" else processed_right
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            import traceback
            print(f"❌ [ERROR] 실시간 동기화 오류: {e}")
            traceback.print_exc()
            time.sleep(0.01)
            continue


# 도우미 함수들
def draw_silhouette_overlay(frame, silhouette, body_parts, colors, results):
    overlay = frame.copy()
    height, width = frame.shape[:2]
    if not results.pose_landmarks:
        return overlay
    landmarks = results.pose_landmarks.landmark
    mismatch_parts = []
    
    for part, indices in body_parts.items():
        part_pixels = silhouette.get(part, np.array([]))
        if part_pixels.shape[0] < 3:
            continue

        part_centroid = np.mean(part_pixels, axis=0) if len(part_pixels) > 0 else None
        landmark_centroid = np.mean([(landmarks[i].x * width, landmarks[i].y * height) for i in indices], axis=0)
        landmark_centroid = np.mean([(landmarks[i].x * width, landmarks[i].y * height) for i in indices], axis=0)
        dist = np.linalg.norm(part_centroid - landmark_centroid)
        color = (0,0,255) if dist > 80 else colors[part]
        pts = part_pixels.reshape(-1,1,2).astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
        return overlay

def update_squat_count(knee_angle):
    global squat_count, is_squatting, warned_low_depth, not_deep_squat, direction , squat_started
    squat_down_threshold = 110  # 내려갔다고 판단하는 각도
    squat_up_threshold = 160    # 완전히 올라왔다고 판단하는 각도
    deep_squat_limit = 99       # 충분히 깊게 내려갔는지 판단
    
    if direction == "up" and knee_angle < squat_down_threshold:
        direction = "down"
        squat_started = True
        # print("⬇️ 내려감 감지")

    elif direction == "down" and knee_angle > squat_up_threshold:
        direction = "up"
        if squat_started:
            if knee_angle < deep_squat_limit:
                not_deep_squat += 1
                print("⚠️ 깊지 않은 스쿼트")
            else:
                squat_count += 1
                print(f"✅ 스쿼트 카운트: {squat_count}")
            squat_started = False  # 다음 사이클로 리셋
    squat_count_status["count"] = squat_count

# ------------------- Flask 앱 초기화 -------------------

app = Flask(__name__, static_folder='static')
app.secret_key = 'your-very-secret-key-1234'  # 세션 사용을 위한 시크릿키 설정


# ------------------- Flask 라우트 -------------------
def generate_stream(camera):
    def generate():
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("⚠️ [WARNING] 프레임이 None입니다. 계속 시도 중...")
                time.sleep(0.1)
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return generate

@app.route('/video_feed/front')
def video_feed_front():
    """
    전면 카메라 스트림을 제공하는 라우트.
    """
    return Response(generate_stream(front_camera)(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/right')
def video_feed_right():
    """
    전면 카메라 스트림을 제공하는 라우트.
    """
    return Response(generate_stream(right_camera)(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/squat_ex2')
def squat_ex2():
    exercise = request.args.get('exercise', 'default_exercise')
    return render_template('squat_ex2.html', exercise=exercise)

@app.route('/squat_guide3')
def squat_guide3():
    # 백그라운드 작업 시작
    thread = Thread(target=process_squat_analysis, args=(front_camera, right_camera))
    thread.start()
    # 실시간 스트림 페이지 렌더링
    return render_template('squat_guide3.html')

pose_analyse_complete = False
silhouette_creation_complete = False
def process_squat_analysis(front_camera, right_camera):
    """
    백그라운드에서 스쿼트 분석 작업을 처리하는 함수.
    """
    global silhouette_creation_complete
    global pose_analyse_complete
    # 1. 프레임 저장
    time.sleep(4) # N초 카운트
    recording_pose(front_camera, right_camera)
    
    def run_pose_analyse():
        global pose_analyse_complete
        analyse_pose()
        data_dict = mk_dictionary()
        st_failed, ht_failed, kd_failed, min_angle_frame = control_squat_posture(data_dict)
        st_accuracy, ht_accuracy, kd_accuracy = show_result(st_failed, ht_failed, kd_failed, min_angle_frame)
        mean_accuracy = round((st_accuracy + ht_accuracy + kd_accuracy) / 3, 1)
        session['mean_accuracy'] = mean_accuracy
        print("✅ [INFO] 분석 작업 완료. squat_check4.html로 이동합니다.")
        pose_analyse_complete = True

    def run_create_silhouettes():
        global silhouette_creation_complete
        create_silhouettes(front_save_image_dir, right_save_image_dir, front_silhouette_dir, right_silhouette_dir)
        silhouette_creation_complete = True
        print("✅ [INFO] 실루엣 생성 완료.")
        global front_silhouette_cache, right_silhouette_cache, data_dict
        data_dict = mk_dictionary()  # CSV 다시 읽기
        front_silhouette_cache = cache_silhouette_npy_with_angle_and_direction(front_silhouette_dir, ["head","left_arm","right_arm","left_leg","right_leg"], data_dict)
        right_silhouette_cache = cache_silhouette_npy_with_angle_and_direction(right_silhouette_dir, ["head","left_arm","right_arm","left_leg","right_leg"], data_dict)
        print("✅ [INFO] 실루엣 캐싱 완료.")

    Thread(target=run_create_silhouettes).start()
   
    Thread(target=run_pose_analyse).start()
    
@app.route('/loading')
def loading():
    """
    로딩 페이지를 렌더링합니다.
    """
    return render_template('loading.html')
    
@app.route('/squat_check4')
def squat_check4():
    global accuracy_result
    return render_template('squat_check4.html',
        st_accuracy=accuracy_result["st_accuracy"],
        ht_accuracy=accuracy_result["ht_accuracy"],
        kd_accuracy=accuracy_result["kd_accuracy"])

@app.route('/setting5', methods=['GET', 'POST'])
def setting5():
    if request.method == 'POST':
        sets = int(request.form.get('sets'))
        reps = int(request.form.get('reps'))
        session['sets'] = sets
        session['reps'] = reps
        session['current_set'] = 1
        return redirect(url_for('squat_start6'))
    return render_template('setting5.html')

@app.route('/squat_start6', methods=['GET', 'POST'])
def squat_start6():
        #global squat_count       
        #squat_count = 0
        current_set = session.get('current_set', 1)
        sets = session.get('sets', 3)
        reps = session.get('reps', 10)
        return render_template('squat_start6.html', sets=sets, reps=reps, current_set=current_set)
    
 # ...existing code...

@app.route('/squat_end7')
def squat_end7():
    # show_result에서 세션에 저장한 평균 정확도 불러오기
    accuracy = session.get('mean_accuracy', None)
    if accuracy is None:
        # 만약 세션에 값이 없으면 임시로 랜덤값 사용 (예외처리)
        accuracy = round(random.uniform(60, 92), 1)

    if accuracy >= 80:
        feedback_list = [
            "아주 잘하고 있어요! 어깨만 조금 더 펴보세요.",
            "좋은 자세입니다! 무릎 방향만 조금 신경 써보세요.",
            "거의 완벽해요! 호흡을 일정하게 해보세요.",
            "멋진 자세예요! 시선을 정면으로 유지해보세요.",
            "잘하고 있습니다! 엉덩이를 조금 더 뒤로 빼보세요."
        ]
    else:
        feedback_list = [
            "조금 더 집중해서 해보세요!",
            "더 깊게 앉아보세요.",
            "자세를 한 번 더 점검해보세요.",
            "꾸준히 연습하면 더 좋아질 거예요.",
            "조금 더 노력해봅시다!"
        ]
    feedback = random.choice(feedback_list)
    return render_template('squat_end7.html', accuracy=accuracy, feedback=feedback)

@app.route('/silhouette_status')
def silhouette_status():
    """
    실루엣 생성 상태를 반환하는 API.
    """
    global silhouette_creation_complete
    return {'complete': silhouette_creation_complete}

@app.route('/pose_analyse_status')
def psoe_analyse_status():
    """
    실루엣 생성 상태를 반환하는 API.
    """
    global pose_analyse_complete
    return {'complete': pose_analyse_complete}

@app.route('/start_stream/<int:camera_index>')
def start_stream(camera_index):
    """Start streaming for the specified camera index."""
    streaming_flags[camera_index] = True
    return '', 204  # Return a no-content response

@app.route('/stop_stream/<int:camera_index>')
def stop_stream(camera_index):
    """Stop streaming for the specified camera index."""
    streaming_flags[camera_index] = False
    return '', 204  # Return a no-content response

@app.route('/video_feed/overlay/front')
def video_feed_overlay_front():
    reps = int(request.args.get('reps', 10))
    sets = int(request.args.get('sets', 3))
    current_set = session.get('current_set', 1)
    return Response(realtime_synchronize("front", data_dict, front_camera, right_camera,reps,sets,current_set),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/overlay/right')
def video_feed_overlay_right():
    reps = int(request.args.get('reps', 10))
    sets = int(request.args.get('sets', 3))
    current_set = session.get('current_set', 1)
    return Response(realtime_synchronize("right", data_dict, front_camera, right_camera,reps,sets,current_set),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_set', methods=['POST'])
def update_set():
    session['current_set'] = int(request.form['current_set'])
    return '', 204

@app.route('/breaktime')
def breaktime():
    global squat_count
    squat_count = 0
    current_set = session.get('current_set', 1)
    sets = session.get('sets', 3)
    # JS에서 30초 후 squat_start6로 자동 이동
    current_set += 1
    session["current_set"] = current_set
    if current_set > sets:
        return redirect(url_for('squat_end7'))
    return render_template('rest.html', current_set=current_set, sets=sets)

@app.route('/squat_count_status')
def squat_count_status_api():
    squat_count_status["set"] = session.get('current_set', 1)
    squat_count_status["sets"] = session.get('sets', 3)
    return jsonify(squat_count_status)

if __name__ == '__main__':
    try:
        clear_folders(front_save_image_dir, right_save_image_dir, front_silhouette_dir, right_silhouette_dir)
        # Flask 서버 실행
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        # 자원 정리
        front_camera.release()
        right_camera.release()
        executor.shutdown(wait=True)
