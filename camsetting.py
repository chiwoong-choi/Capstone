import cv2

def find_connected_cameras(max_devices=10):
    available_cameras = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"No camera at index {i}")
    return available_cameras

if __name__ == "__main__":
    cameras = find_connected_cameras()
    print(f"Available camera indexes: {cameras}")
