import cv2
import socket
import struct
import time
import numpy as np 
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

# 카메라별 프레임 카운터
front_frame_counter = 0
right_frame_counter = 0

def get_counter_timestamp(counter):
    return counter * 10000  # 1만 단위로 증가

def send_frame(sock, frame, timestamp):
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    size = len(data)
    header = struct.pack("QI", timestamp, size)
    sock.sendall(header + data)

try:
    while True:
        # 전면 카메라
        ret_front, frame_front = front_camera.read()
        if ret_front:
            ts_front = get_counter_timestamp(front_frame_counter)
            send_frame(front_socket, frame_front, ts_front)
        else:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            ts_front = get_counter_timestamp(front_frame_counter)
            send_frame(front_socket, dummy, ts_front)
        front_frame_counter += 1

        # 측면 카메라 (right)도 동일하게 처리
        ret_right, frame_right = right_camera.read()
        if ret_right:
            ts_right = get_counter_timestamp(right_frame_counter)
            send_frame(right_socket, frame_right, ts_right)
        else:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            ts_right = get_counter_timestamp(right_frame_counter)
            send_frame(right_socket, dummy, ts_right)
        right_frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)  # 너무 빠르게 보내지 않도록 조절

finally:
    front_camera.release()
    right_camera.release()
    front_socket.close()
    right_socket.close()
    print("✅ [INFO] 전송 종료 및 자원 정리 완료.")
