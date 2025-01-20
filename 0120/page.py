from flask import Flask, request, render_template, Response
import cv2
import mediapipe as mp
import time


app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 카메라 스트림 연결
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise Exception("카메라 스트림을 열 수 없습니다.")

frame_count = 0
start_time = time.time()
frame_skip = 2  # 성능 최적화를 위해 n번째 프레임만 처리

def generate_frames():
    global frame_count, start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 가져올 수 없습니다.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # 프레임 크기 조정
        frame = cv2.resize(frame, (640, 480)) 

        # BGR에서 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Pose 처리
        results = pose.process(rgb_frame)

        # 포즈 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # FPS 계산
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 스트림 데이터 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exercise.html', methods=['GET'])
def exercise():
    exercise_type = request.args.get('exercise', 'none')
    return render_template('exercise.html', exercise=exercise_type)

if __name__ == "__main__":
    app.run(debug=True)