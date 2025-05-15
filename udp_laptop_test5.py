#test_9와 연동
import cv2
import socket
import struct
import time

# UDP 소켓 설정
FRONT_CAMERA_PORT = 5005  # 전면 카메라 포트
RIGHT_CAMERA_PORT = 5006  # 측면 카메라 포트
TCP_IP = "223.194.136.87"  # 노트북 IP 주소

# 두 개의 소켓 생성
front_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
right_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

front_socket.connect((TCP_IP, FRONT_CAMERA_PORT))
right_socket.connect((TCP_IP, RIGHT_CAMERA_PORT))

# 카메라 초기화
front_camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 전면 카메라
right_camera = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # 측면 카메라

if not front_camera.isOpened() or not right_camera.isOpened():
    print("⚠️ [ERROR] 카메라를 열 수 없습니다.")
    exit()

last_timestamp_front = 0
last_timestamp_right = 0

def get_monotonic_timestamp(last_timestamp):
    now = int(time.time() * 1e6)
    if now <= last_timestamp:
        now = last_timestamp + 1
    return now

def send_frame(socket, frame, last_timestamp):
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    size = len(data)
    timestamp = get_monotonic_timestamp(last_timestamp)
    header = struct.pack("QI", timestamp, size)
    socket.sendall(header + data)
    return timestamp

while True:
    # 전면 카메라 프레임 읽기
    ret_front, frame_front = front_camera.read()
    if ret_front:
        last_timestamp_front = send_frame(front_socket, frame_front, last_timestamp_front)

    # 측면 카메라 프레임 읽기
    ret_side, frame_side = right_camera.read()
    if ret_side:
        last_timestamp_right = send_frame(right_socket, frame_side, last_timestamp_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

front_camera.release()
right_camera.release()
front_socket.close()
right_socket.close()