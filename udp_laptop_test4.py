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
front_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

front_socket.connect((TCP_IP, FRONT_CAMERA_PORT))
right_socket.connect((TCP_IP, RIGHT_CAMERA_PORT))

# 카메라 초기화
front_camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 전면 카메라
right_camera = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # 측면 카메라

if not front_camera.isOpened() or not right_camera.isOpened():
    print("⚠️ [ERROR] 카메라를 열 수 없습니다.")
    exit()

# 패킷 크기 제한
def send_frame(socket, frame):
    """
    프레임을 TCP로 전송하는 함수.
    """
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    size = len(data)
    timestamp = int(time.time() * 1e6)  # 마이크로초 단위 타임스탬프
    header = struct.pack("QI", timestamp, size)  # 타임스탬프(8바이트) + 데이터 크기(4바이트)
    socket.sendall(header + data)

while True:
    # 전면 카메라 프레임 읽기
    ret_front, frame_front = front_camera.read()
    if ret_front:
        send_frame(front_socket, frame_front)

    # 측면 카메라 프레임 읽기
    ret_side, frame_side = right_camera.read()
    if ret_side:
        send_frame(right_socket, frame_side)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

front_camera.release()
right_camera.release()
front_socket.close()
right_socket.close()