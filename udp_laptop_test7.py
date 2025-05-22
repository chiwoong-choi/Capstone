import cv2
import socket
import struct
import time

# TCP 연결 설정
FRONT_CAMERA_PORT = 5005
RIGHT_CAMERA_PORT = 5006
TCP_IP = "192.168.111.60"  # 수신 서버의 IP 주소

# 소켓 설정
front_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
right_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

front_socket.connect((TCP_IP, FRONT_CAMERA_PORT))
right_socket.connect((TCP_IP, RIGHT_CAMERA_PORT))

# 카메라 초기화
front_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
right_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not front_camera.isOpened() or not right_camera.isOpened():
    print("⚠️ [ERROR] 카메라를 열 수 없습니다.")
    exit()

# 카메라별 타임스탬프 독립 관리
last_timestamp_front = 0
last_timestamp_right = 0

def get_monotonic_timestamp(camera):
    """
    카메라별로 순차적으로 증가하는 타임스탬프를 생성합니다.
    """
    global last_timestamp_front, last_timestamp_right
    new_ts = int(time.monotonic() * 1e6)  # 마이크로초 단위

    if camera == 'front':
        if new_ts <= last_timestamp_front:
            new_ts = last_timestamp_front + 1
        last_timestamp_front = new_ts
    elif camera == 'right':
        if new_ts <= last_timestamp_right:
            new_ts = last_timestamp_right + 1
        last_timestamp_right = new_ts

    return new_ts

def send_frame(sock, frame, timestamp):
    """
    소켓을 통해 프레임 전송
    """
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    size = len(data)
    header = struct.pack("QI", timestamp, size)
    sock.sendall(header + data)

# 메인 루프
try:
    while True:
        ret_front, frame_front = front_camera.read()
        if ret_front:
            ts_front = get_monotonic_timestamp('front')
            send_frame(front_socket, frame_front, ts_front)

        ret_right, frame_right = right_camera.read()
        if ret_right:
            ts_right = get_monotonic_timestamp('right')
            send_frame(right_socket, frame_right, ts_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)  # 너무 빠르게 보내지 않도록 조절

finally:
    front_camera.release()
    right_camera.release()
    front_socket.close()
    right_socket.close()
    print("✅ [INFO] 전송 종료 및 자원 정리 완료.")

