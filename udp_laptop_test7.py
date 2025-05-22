import cv2
import socket
import struct
import time

FRONT_CAMERA_PORT = 5005
RIGHT_CAMERA_PORT = 5006
TCP_IP = "192.168.111.60"

front_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
right_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

front_socket.connect((TCP_IP, FRONT_CAMERA_PORT))
right_socket.connect((TCP_IP, RIGHT_CAMERA_PORT))

front_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
right_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not front_camera.isOpened() or not right_camera.isOpened():
    print("⚠️ [ERROR] 카메라를 열 수 없습니다.")
    exit()

last_timestamp = 0

def get_monotonic_timestamp():
    global last_timestamp
    new_ts = int(time.monotonic() * 1e6)
    if new_ts <= last_timestamp:
        new_ts = last_timestamp + 1
    last_timestamp = new_ts
    return new_ts

def send_frame(sock, frame):
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()
    size = len(data)
    timestamp = get_monotonic_timestamp()
    header = struct.pack("QI", timestamp, size)
    sock.sendall(header + data)

while True:
    ret_front, frame_front = front_camera.read()
    if ret_front:
        send_frame(front_socket, frame_front)

    ret_right, frame_right = right_camera.read()
    if ret_right:
        send_frame(right_socket, frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.05)

front_camera.release()
right_camera.release()
front_socket.close()
right_socket.close()
