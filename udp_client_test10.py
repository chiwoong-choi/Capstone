#틀린거 경고 추가
#수정 내용: generate_silhouette()함수에서 편입된 픽셀의 좌표를 npy파일로 저장
#저장된 파일을 realtime_synchronize()함수에서 불러와서 사용
#랜드마크 거리에 따라 실루엣과의 차이 계산 및 색상 경고 추가
#수정 필요 사항: mediapipe 연산을 위한 타임스탬프 -> realtime_synchronize함수의 pose객체 주기적 초기화
import cv2
import os
import time
import csv
import numpy as np
import socket
from flask import Flask, render_template, send_from_directory, Response, request, redirect, url_for, abort
from threading import Thread, Lock
import mediapipe as mp
import struct
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

#ffmpeg 인코딩버전

# ------------------- UDPFrameReceiver 클래스 -------------------
import threading

class TCPFrameReceiver:
    def __init__(self, tcp_ip, tcp_port):
        """
        TCP 소켓을 통해 프레임을 수신하는 클래스.
        
        Args:
            tcp_ip (str): 수신할 IP 주소.
            tcp_port (int): 수신할 포트 번호.
        """
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((tcp_ip, tcp_port))
        self.socket.listen(1)  # 클라이언트 연결 대기
        print(f"✅ [INFO] TCP 서버가 {tcp_ip}:{tcp_port}에서 대기 중입니다...")
        
        self.conn, self.addr = self.socket.accept()  # 클라이언트 연결 수락
        print(f"✅ [INFO] 클라이언트 연결 수락: {self.addr}")
        
        self.running = True
        self.frame = None
        #self.timestamp = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """
        TCP 소켓을 통해 지속적으로 프레임을 수신하는 메서드.
        """
        while self.running:
            try:
                
                # 데이터 크기 헤더(4바이트) 수신
                header = self.conn.recv(12)
                
                if not header:
                    print("⚠️ [WARNING] 헤더를 수신하지 못했습니다. 연결이 종료되었을 수 있습니다.")
                    break
                timestamp, size = struct.unpack("QI", header)

                # 프레임 데이터 수신
                data = b""
                while len(data) < size:
                    packet = self.conn.recv(size - len(data))
                    if not packet:
                        print("⚠️ [WARNING] 데이터 수신 중 연결이 종료되었습니다.")
                        break
                    data += packet

                # 프레임 디코딩
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        if hasattr(self, 'timestamp') and self.timestamp is not None:
                            if timestamp <= self.timestamp:
                                #print(f"⚠️ [WARNING] 역전된 timestamp 프레임 무시: {timestamp} <= {self.timestamp}")
                                continue
                        self.frame = frame
                        self.timestamp = timestamp
                        #print("✅ [INFO] 프레임 수신 및 디코딩 완료")
            except Exception as e:
                print(f"⚠️ [WARNING] 프레임 수신 중 오류 발생: {e}")
                break

    def get_frame(self):
        """
        최신 프레임을 반환하는 메서드.
        
        Returns:
            np.ndarray: 최신 프레임 (없으면 None).
        """
        with self.lock:
            if self.frame is None:
                print("⚠️ [WARNING] 프레임이 None입니다. 데이터를 수신하지 못했습니다.")
                return None, None
        return self.frame.copy(), self.timestamp

    def stop(self):
        """
        수신을 중지하고 소켓 및 스레드를 정리하는 메서드.
        """
        self.running = False
        self.thread.join()
        self.conn.close()
        self.socket.close()
        print("🔴 [INFO] TCP 서버가 종료되었습니다.")

# ------------------- 동기화 함수 -------------------
def get_synchronized_frames(front_camera, right_camera, max_time_diff=200000):
    """
    두 카메라의 최신 프레임을 동기화하여 반환합니다.
    
    Args:
        front_camera (TCPFrameReceiver): 전면 카메라 수신기.
        right_camera (TCPFrameReceiver): 측면 카메라 수신기.
        max_time_diff (int): 두 프레임의 타임스탬프 차이 허용 범위 (마이크로초 단위).
    
    Returns:
        tuple: 동기화된 전면 및 측면 프레임 (없으면 None, None).
    """
    front_frame, front_timestamp = front_camera.get_frame()
    right_frame, right_timestamp = right_camera.get_frame()

    if front_frame is None or right_frame is None:
        print("⚠️ [WARNING] 동기화된 프레임을 가져올 수 없습니다.")
        return None, None

    # 타임스탬프 차이 계산
    time_diff = abs(front_timestamp - right_timestamp)
    if time_diff > max_time_diff:
        print(f"⚠️ [WARNING] 타임스탬프 차이가 너무 큽니다: {time_diff}μs")
        return None, None

    return front_frame, right_frame

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

os.makedirs(front_save_image_dir, exist_ok=True)
os.makedirs(right_save_image_dir, exist_ok=True)

result_save_image_dir = "frames"
front_result_output_video_path = "front_result.mp4"
right_result_output_video_path = "right_result.mp4"

streaming_flags = {0: False, 1: False}
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
        front_frame, right_frame = get_synchronized_frames(front_camera, right_camera)
        
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
    front_frame_files = sorted(os.listdir(front_save_image_dir))  # 정렬하여 순서대로 재생
    right_frame_files = sorted(os.listdir(right_save_image_dir))
    
    st_length = len(unique_st_failed_front_frame_num)
    ht_length = len(unique_ht_failed_front_frame_num)
    kd_length = len(unique_kd_failed_front_frame_num)
    
    total_frames = len(front_frame_files)  # 총 프레임 수
    
    st_accuracy = (total_frames - st_length) / total_frames * 100
    ht_accuracy = (total_frames - ht_length) / total_frames * 100
    kd_accuracy = (total_frames - kd_length) / total_frames * 100
    
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
    return st_accuracy, ht_accuracy, kd_accuracy

# ------------------- 모범자세 실루엣 생성  -------------------
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

def process_frame(camera_type, frame, timestamp, pose, silhouette_dir, silhouette_angle, body_parts, colors, previous_knee_angle=None, previous_silhouette=None, angle_threshold=1):
    """
    단일 프레임을 처리하는 함수.
    실루엣 오버레이 최적화 로직 포함.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)#, timestamp=timestamp)

    if not results.pose_landmarks:
        return frame, previous_knee_angle, previous_silhouette  # 랜드마크가 없으면 원본 프레임 반환

    landmarks = results.pose_landmarks.landmark
    height, width, _ = frame.shape

    # 무릎 각도 계산
    hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height)
    knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * height)
    ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height)
    knee_angle = calculate_angle_knee(hip, knee, ankle)
    
    # 다리 각도 변화가 거의 없는 경우 이전 실루엣 사용
    if previous_knee_angle is not None and abs(knee_angle - previous_knee_angle) <= angle_threshold:
        silhouette = previous_silhouette
    else:
        # 실루엣 선택
        closest_frame_index = np.argmin(np.abs(silhouette_angle - knee_angle))
        silhouette = {}
        for part in body_parts:
            try:
                silhouette[part] = np.load(os.path.join(silhouette_dir, f"{closest_frame_index:04d}_{part}.npy"))
            except Exception as e:
                print(f"⚠️ [WARNING] {camera_type} - {part} 실루엣 로드 실패: {e}")
                silhouette[part] = []

        # 이전 실루엣 업데이트
        previous_silhouette = silhouette

    # 이전 다리 각도 업데이트
    previous_knee_angle = knee_angle

    # 실루엣 오버레이
    overlay = frame.copy()
    mismatch_parts = []

    for part, indices in body_parts.items():
        try:
            part_pixels = silhouette.get(part, [])
            part_centroid = np.mean(part_pixels, axis=0) if len(part_pixels) > 0 else None
            landmark_centroid = np.mean([(landmarks[i].x * width, landmarks[i].y * height) for i in indices], axis=0)
            if part_centroid is not None and landmark_centroid is not None:
                dist = np.linalg.norm(np.array(part_centroid) - np.array(landmark_centroid))
                if dist > 80:  # 오차 허용 기준(px)
                    mismatch_parts.append(part)
            for pt in part_pixels:
                color = (0, 0, 255) if part in mismatch_parts else colors[part]
                cv2.circle(overlay, tuple(pt), 1, color, -1)
        except Exception as e:
            print(f"⚠️ [WARNING] {camera_type} - {part} 오버레이 실패: {e}")
            continue
        
    
    
    return overlay, previous_knee_angle, previous_silhouette

def realtime_synchronize(camera_type, data_dict, front_camera_thread, right_camera_thread,reps=1, sets=3):
    """
    전면 및 측면 카메라를 병렬로 처리하여 스트리밍.
    """
    print("🔵 [INFO] 실루엣 동기화 시작")

    silhouette_angle = np.array([value[2] for value in data_dict.values()])  # 각 프레임의 무릎 각도
    colors = {
        "head": (255, 0, 0),
        "left_arm": (255, 0, 0),
        "right_arm": (255, 0, 0),
        "left_leg": (255, 0, 0),
        "right_leg": (255, 0, 0)
        #torso": (255, 0, 0)
    }
    body_parts = {
        "head": [0],
        "left_arm": [11, 13, 15],
        "right_arm": [12, 14, 16],
        "left_leg": [23, 25, 27],
        "right_leg": [24, 26, 28]
        #"torso": [11, 12, 23, 24]
    }

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # 실루엣 오버랩 최적화 변수
    previous_knee_angle_front = None
    previous_silhouette_front = None
    previous_knee_angle_right = None
    previous_silhouette_right = None
    squat_count = 0
    is_squatting = False
    sqaut_count_setting = reps
    set_count_setting = sets
    
    while True:
        try:
            if executor._shutdown:
                print("❌ [ERROR] executor가 이미 종료되었습니다. 스트림 중단.")
                break

        
            
            # 전면 및 측면 프레임 가져오기
            front_frame, front_timestamp = front_camera_thread.get_frame()
            right_frame, right_timestamp = right_camera_thread.get_frame()

            if front_frame is None or right_frame is None:
                print("⚠️ [WARNING] 프레임 없음. 종료.")
                break

            # 병렬 처리
            future_front = executor.submit(
                process_frame, "front", front_frame, front_timestamp, pose, front_silhouette_dir,
                silhouette_angle, body_parts, colors, previous_knee_angle_front, previous_silhouette_front
            )
            future_right = executor.submit(
                process_frame, "right", right_frame, right_timestamp, pose, right_silhouette_dir,
                silhouette_angle, body_parts, colors, previous_knee_angle_right, previous_silhouette_right
            )

            processed_front, previous_knee_angle_front, previous_silhouette_front = future_front.result()
            processed_right, previous_knee_angle_right, previous_silhouette_right = future_right.result()

            knee_angle = previous_knee_angle_front
            if knee_angle is not None:
                if knee_angle > 140:
                    if is_squatting:
                        squat_count += 1
                        print(f"✅ [INFO] 스쿼트 카운트: {squat_count}")
                        is_squatting = False
                elif knee_angle < 100:
                    is_squatting = True
                    
            # 결과 스트리밍
            if camera_type == "front":
                _, buffer = cv2.imencode('.jpg', processed_front)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                if squat_count >= sqaut_count_setting:
                    print("[INFO] 10회 이상 스쿼트 완료")
                    break
            
            elif camera_type == "right":
                _, buffer = cv2.imencode('.jpg', processed_right)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
                    
            
        except Exception as e:
            print(f"❌ [ERROR] 실시간 동기화 중 오류 발생: {e}")
            continue
    
# ------------------- Flask 앱 초기화 -------------------

app = Flask(__name__, static_folder='static')

# 동기화된 프레임 저장
front_camera = TCPFrameReceiver("0.0.0.0", 5005)  # 전면 카메라 포트
right_camera = TCPFrameReceiver("0.0.0.0", 5006)  # 측면 카메라 포트



# ------------------- Flask 라우트 -------------------
def generate_stream(camera):
    def generate():
        while True:
            frame_tuple = camera.get_frame()
            if frame_tuple is None or frame_tuple[0] is None:
                print("⚠️ [WARNING] 프레임이 None입니다. 계속 시도 중...")
                time.sleep(0.1)
                continue
            
            frame = frame_tuple[0]  # 프레임만 추출
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
        show_result(st_failed, ht_failed, kd_failed, min_angle_frame)
        print("✅ [INFO] 분석 작업 완료. squat_check4.html로 이동합니다.")
        pose_analyse_complete = True
    def run_create_silhouettes():
        global silhouette_creation_complete
        create_silhouettes(front_save_image_dir, right_save_image_dir, front_silhouette_dir, right_silhouette_dir)
        silhouette_creation_complete = True
        print("✅ [INFO] 실루엣 생성 완료.")
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
    return render_template('squat_check4.html')
    
@app.route('/setting5', methods=['GET', 'POST'])
def setting5():
    if request.method == 'POST':
        print("POST 요청 수신. squat_start6로 리다이렉트합니다.")
        return redirect(url_for('squat_start6'))
    print("GET 요청 수신. setting5.html 렌더링합니다.")
    return render_template('setting5.html')

@app.route('/squat_start6')
def squat_start6():
    sets = request.args.get('sets', 3)
    reps = request.args.get('reps', 10)
    return render_template('squat_start6.html', sets=sets, reps=reps)

@app.route('/squat_end7')
def squat_end7():
    accuracy = round(random.uniform(80, 100), 2)  # 무작위 정확도 생성
    feedback = "무릎을 더 벌리세요."
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
    return Response(realtime_synchronize("front", data_dict, front_camera, right_camera,reps,sets),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/overlay/right')
def video_feed_overlay_right():
    reps = int(request.args.get('reps', 10))
    sets = int(request.args.get('sets', 3))
    return Response(realtime_synchronize("right", data_dict, front_camera, right_camera,reps,sets),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        clear_folders(front_save_image_dir, right_save_image_dir, front_silhouette_dir, right_silhouette_dir)
        # Flask 서버 실행
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        # 자원 정리
        front_camera.stop()
        right_camera.stop()
        executor.shutdown(wait=True)
