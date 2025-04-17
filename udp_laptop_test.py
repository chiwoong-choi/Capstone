import cv2
import socket
import struct
import time

# UDP 소켓 설정
FRONT_CAMERA_PORT = 5005  # 전면 카메라 포트
RIGHT_CAMERA_PORT = 5006  # 측면 카메라 포트
UDP_IP = "223.194.137.232"  # 노트북 IP 주소

# 두 개의 소켓 생성
front_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 초기화
front_camera = cv2.VideoCapture(0)  # 전면 카메라
right_camera = cv2.VideoCapture(1)  # 측면 카메라

if not front_camera.isOpened() or not right_camera.isOpened():
    print("⚠️ [ERROR] 카메라를 열 수 없습니다.")
    exit()

while True:
    # 전면 카메라 프레임 읽기
    ret_front, frame_front = front_camera.read()
    if ret_front:
        _, buffer_front = cv2.imencode('.jpg', frame_front)
        timestamp = time.time()
        data = struct.pack("d", timestamp) + buffer_front.tobytes()
        front_socket.sendto(data, (UDP_IP, FRONT_CAMERA_PORT))

    # 측면 카메라 프레임 읽기
    ret_side, frame_side = right_camera.read()
    if ret_side:
        _, buffer_side = cv2.imencode('.jpg', frame_side)
        timestamp = time.time()
        data = struct.pack("d", timestamp) + buffer_side.tobytes()
        right_socket.sendto(data, (UDP_IP, RIGHT_CAMERA_PORT))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
front_camera.release()
right_camera.release()
front_socket.close()
right_socket.close()

