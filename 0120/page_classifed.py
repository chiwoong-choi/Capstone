from flask import Flask, request, render_template, Response
import cv2
import mediapipe as mp
import time

class PoseEstimationApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.pose = self.initialize_pose_model()
        self.cap = self.initialize_camera()
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_skip = 2  # 성능 최적화를 위해 n번째 프레임만 처리

        # 라우트 설정
        self.setup_routes()

    def initialize_pose_model(self):
        """MediaPipe Pose 모델 초기화"""
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        return pose

    def initialize_camera(self):
        """카메라 스트림 초기화"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("카메라 스트림을 열 수 없습니다.")
        return cap

    def generate_frames(self):
        """웹캠 프레임을 처리 및 스트리밍"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("카메라에서 프레임을 가져올 수 없습니다.")
                break

            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))  # 프레임 크기 조정
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
            results = self.pose.process(rgb_frame)  # 포즈 추적 수행

            # 랜드마크 그리기
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # FPS 계산 및 출력
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 프레임을 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def setup_routes(self):
        """Flask 라우트 설정"""

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/exercise.html', methods=['GET'])
        def exercise():
            exercise_type = request.args.get('exercise', 'none')
            return render_template('exercise.html', exercise=exercise_type)

    def run(self):
        """Flask 애플리케이션 실행"""
        self.app.run(debug=True)

if __name__ == "__main__":
    app = PoseEstimationApp()
    app.run()