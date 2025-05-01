import cv2
import socket
import struct
import time

# UDP 소켓 설정
FRONT_CAMERA_PORT = 5005  # 전면 카메라 포트
RIGHT_CAMERA_PORT = 5006  # 측면 카메라 포트
UDP_IP = "223.194.131.207"  # 노트북 IP 주소

# 두 개의 소켓 생성
front_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
right_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 초기화
front_camera = cv2.VideoCapture(0)  # 전면 카메라
right_camera = cv2.VideoCapture(1)  # 측면 카메라

if not front_camera.isOpened() or not right_camera.isOpened():
    print("⚠️ [ERROR] 카메라를 열 수 없습니다.")
    exit()

# 패킷 크기 제한
MAX_PACKET_SIZE = 1400  # 1,400 바이트 (헤더 포함)

def send_frame(socket, frame, address, port):
    """
    프레임을 잘게 나누어 UDP로 전송하는 함수.
    """
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    total_size = len(data)
    num_packets = (total_size // MAX_PACKET_SIZE) + 1

    for i in range(num_packets):
        start = i * MAX_PACKET_SIZE
        end = start + MAX_PACKET_SIZE
        packet_data = data[start:end]
        header = struct.pack("H", i) + struct.pack("H", num_packets)  # 패킷 번호와 총 패킷 수
        socket.sendto(header + packet_data, (address, port))

while True:
    # 전면 카메라 프레임 읽기
    ret_front, frame_front = front_camera.read()
    if ret_front:
        send_frame(front_socket, frame_front, UDP_IP, FRONT_CAMERA_PORT)

    # 측면 카메라 프레임 읽기
    ret_side, frame_side = right_camera.read()
    if ret_side:
        send_frame(right_socket, frame_side, UDP_IP, RIGHT_CAMERA_PORT)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

front_camera.release()
right_camera.release()
front_socket.close()
right_socket.close()