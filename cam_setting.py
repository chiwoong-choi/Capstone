import cv2
import threading
import time

class CameraThread:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
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
            time.sleep(0.01)  # CPU 점유율을 너무 높이지 않기 위해 잠시 대기

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            
            # 크롭할 영역 설정 (예: 중앙 영역만 남기기)
            x_start, x_end = 250, 490   # 가로 크롭 (640px 중 중앙 부분)
            y_start, y_end = 50, 430    # 세로 크롭 (480px 중 적절한 부분)

            cropped_frame = self.frame[y_start:y_end, x_start:x_end]
            return cropped_frame.copy() if cropped_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# 카메라 2대 사용
front_camera = CameraThread(0)
right_camera = CameraThread(1)

while True:
    frame_front = front_camera.get_frame()
    frame_right = right_camera.get_frame()

    if frame_front is not None and frame_right is not None:
        cv2.imshow("Front Camera (Cropped)", frame_front)
        cv2.imshow("Right Camera (Cropped)", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
front_camera.stop()
right_camera.stop()
cv2.destroyAllWindows()
