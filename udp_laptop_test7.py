#test_9와 연동
import cv2
import socket
import struct
import time

# UDP 소켓 설정
FRONT_CAMERA_PORT = 5005  # 전면 카메라 포트
RIGHT_CAMERA_PORT = 5006  # 측면 카메라 포트
TCP_IP = "223.194.129.240"  # 노트북 IP 주소

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

#front_frame_counter = 0
#right_frame_counter = 0

def get_monotonic_timestamp():
    return int(time.monotonic() * 1e6)

def send_frame(socket, frame):
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    size = len(data)
    timestamp = get_monotonic_timestamp()
    header = struct.pack("QI", timestamp, size)
    socket.sendall(header + data)
    

while True:
    # 전면 카메라 프레임 읽기
    ret_front, frame_front = front_camera.read()
    if ret_front:
        front_frame_counter = send_frame(front_socket, frame_front)
    
    # 측면 카메라 프레임 읽기
    ret_side, frame_side = right_camera.read()
    if ret_side:
        right_frame_counter = send_frame(right_socket, frame_side)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.05)  # 10ms 딜레이 추가
    
front_camera.release()
right_camera.release()
front_socket.close()
right_socket.close()