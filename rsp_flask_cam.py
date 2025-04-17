import cv2
import os
import time
import csv
import numpy as np
from flask import Flask, Response
from threading import Thread, Lock
import mediapipe as mp

# ------------------- 포즈 유틸리티 -------------------
def calculate_angle_knee(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# ------------------- 카메라 스레드 클래스 -------------------
class CameraThread:
    def __init__(self, camera_id):
        print(f"[INFO] 카메라 {camera_id} 초기화 중...")
        self.cap = cv2.VideoCapture(camera_id)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"[ERROR] Camera {camera_id} 열기 실패")

        self.running = True
        self.frame = None
        self.lock = Lock()
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        print(f"[INFO] 카메라 {camera_id} 스레드 시작 완료")

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        print(f"[INFO] 카메라 {camera_id} 정지됨")

# ------------------- CSV에서 data_dict 로드 -------------------
def load_data_dict_from_csv(csv_path="squat_analysis.csv"):
    print(f"[INFO] CSV 파일 '{csv_path}' 로드 중...")
    data_dict = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        for row in reader:
            if len(row) != 6:
                continue
            filename = row[0]
            values = list(map(float, row[1:]))
            data_dict[filename] = values
    print(f"[INFO] 총 {len(data_dict)}개의 프레임 데이터를 불러옴")
    return data_dict

# ------------------- 오버레이 생성기 -------------------
def generate_overlay_stream(data_dict, front_camera, right_camera):
    print("[INFO] Overlay 스트리밍 시작")

    print("[INFO] 실루엣 이미지 로딩 중...")
    front_silhouettes = [cv2.imread(os.path.join("front_silhouette", f)) for f in sorted(os.listdir("front_squat_frames"))]
    right_silhouettes = [cv2.imread(os.path.join("right_silhouette", f)) for f in sorted(os.listdir("right_squat_frames"))]
    print(f"[INFO] front_silhouettes: {len(front_silhouettes)}개, right_silhouettes: {len(right_silhouettes)}개 로드됨")

    silhouette_angle = np.array([value[2] for value in data_dict.values()])
    previous_knee_angle = None
    previous_front_silhouette = None
    previous_right_silhouette = None
    angle_threshold = 1
    frame_counter = 0
    process_interval = 5

    while True:
        frame_front = front_camera.get_frame()
        frame_right = right_camera.get_frame()

        if frame_front is None or frame_right is None:
            continue

        frame_front_rgb = cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB)
        frame_counter += 1

        if frame_counter % process_interval == 0:
            results_front = pose.process(frame_front_rgb)
        else:
            results_front = None

        if results_front and results_front.pose_landmarks:
            landmarks = results_front.pose_landmarks.landmark
            hip = (landmarks[24].x, landmarks[24].y)
            knee = (landmarks[26].x, landmarks[26].y)
            ankle = (landmarks[28].x, landmarks[28].y)
            knee_angle = calculate_angle_knee(hip, knee, ankle)
            print(f"[INFO] 현재 프레임 무릎 각도: {knee_angle:.2f}")

            if previous_knee_angle is not None and abs(knee_angle - previous_knee_angle) <= angle_threshold:
                front_silhouette = previous_front_silhouette
                right_silhouette = previous_right_silhouette
            else:
                closest_frame_index = np.argmin(np.abs(silhouette_angle - knee_angle))
                print(f"[INFO] 가장 유사한 실루엣 프레임 index: {closest_frame_index}")
                front_silhouette = front_silhouettes[closest_frame_index]
                right_silhouette = right_silhouettes[closest_frame_index]
                previous_front_silhouette = front_silhouette
                previous_right_silhouette = right_silhouette

            previous_knee_angle = knee_angle

            alpha, beta = 0.7, 0.3
            front_overlay = (alpha * frame_front + beta * front_silhouette).astype(np.uint8)
            right_overlay = (alpha * frame_right + beta * right_silhouette).astype(np.uint8)

            combined = np.hstack((front_overlay, right_overlay))
            _, buffer = cv2.imencode('.jpg', combined)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ------------------- Flask 웹 서버 -------------------
app = Flask(__name__)
print("[INFO] Flask 서버 초기화 중...")
front_cam = CameraThread(0)
right_cam = CameraThread(1)
data_dict = load_data_dict_from_csv("squat_analysis.csv")

@app.route('/')
def index():
    return '<h1>스쿼트 실루엣 오버레이 스트리밍</h1><img src="/overlay">'

@app.route('/overlay')
def overlay():
    print("[INFO] /overlay 요청 수신됨 - 스트리밍 시작")
    return Response(generate_overlay_stream(data_dict, front_cam, right_cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("[INFO] Flask 앱 실행 시작")
    app.run(host='0.0.0.0', port=5000, threaded=True)
