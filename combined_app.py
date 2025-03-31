from flask import Flask, render_template, send_from_directory, Response, request, redirect, url_for, abort
import os
import cv2
import numpy as np
import time
import csv
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="static")

# Directories for saving frames and results
front_save_image_dir = "front_squat_frames"
right_save_image_dir = "right_squat_frames"
os.makedirs(front_save_image_dir, exist_ok=True)
os.makedirs(right_save_image_dir, exist_ok=True)

# Video paths for results
front_result_output_video_path = "static/videos/front_result.mp4"
right_result_output_video_path = "static/videos/right_result.mp4"

# Flags for webcam streaming
streaming_flags = {0: False, 1: False}

# Helper functions for pose analysis
def calculate_angle_knee(a, b, c):
    vector_ab = np.array([a[0] - b[0], a[1] - b[1]])
    vector_bc = np.array([c[0] - b[0], c[1] - b[1]])
    dot_product = np.dot(vector_ab, vector_bc)
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_bc = np.linalg.norm(vector_bc)
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def calculate_tilt_hip(hip_left, hip_right):
    hip_vect = np.array([hip_right[0] - hip_left[0], hip_right[1] - hip_left[1]])
    ground_vect = np.array([1, 0])
    dot_product_hip = np.dot(hip_vect, ground_vect)
    magnitude_hip_vect = np.linalg.norm(hip_vect)
    cos_hip = dot_product_hip / magnitude_hip_vect
    angle_hip = np.degrees(np.arccos(np.clip(cos_hip, -1.0, 1.0)))
    return angle_hip

def calculate_tilt_shoulder(shoulder_left, shoulder_right):
    shoulder_vect = np.array([shoulder_right[0] - shoulder_left[0], shoulder_right[1] - shoulder_left[1]])
    ground_vect = np.array([1, 0])
    dot_product_shoulder = np.dot(shoulder_vect, ground_vect)
    magnitude_shoulder_vect = np.linalg.norm(shoulder_vect)
    cos_shoulder = dot_product_shoulder / magnitude_shoulder_vect
    angle_shoulder = np.degrees(np.arccos(np.clip(cos_shoulder, -1.0, 1.0)))
    return angle_shoulder

def calculate_knee_dis(knee_left, knee_right):
    dis_knee = np.abs(knee_left[0] - knee_right[0])
    return dis_knee

def calculate_shoulder_dis(shoulder_left, shoulder_right):
    dis_shoulder = np.abs(shoulder_left[0] - shoulder_right[0])
    return dis_shoulder

def recording_pose():
    front_frame_count = 0
    right_frame_count = 0
    recording_start_time = time.time()

    while time.time() - recording_start_time < 5:
        front_ret, front_frame = cv2.VideoCapture(0).read()
        right_ret, right_frame = cv2.VideoCapture(1).read()

        if not front_ret or not right_ret:
            print("Cannot fetch frames from cameras.")
            break

        front_frame = cv2.resize(front_frame, (640, 480))
        right_frame = cv2.resize(right_frame, (640, 480))
        front_frame_filename = os.path.join(front_save_image_dir, f"{front_frame_count:04d}.png")
        right_frame_filename = os.path.join(right_save_image_dir, f"{right_frame_count:04d}.png")
        cv2.imwrite(front_frame_filename, front_frame)
        cv2.imwrite(right_frame_filename, right_frame)
        front_frame_count += 1
        right_frame_count += 1

        cv2.imshow('Camera 1 - Front Squat', front_frame)
        cv2.imshow('Camera 2 - Right Squat', right_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Recording complete. Frames saved.")

def analyse_pose():
    with open("squat_analysis.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame File", "Knee Distance", "Shoulder Tilt", "Knee Angle", "Hip Tilt"])

        front_frame_files = sorted(os.listdir(front_save_image_dir))

        for frame_file in front_frame_files:
            frame_path = os.path.join(front_save_image_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                hip_left = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                hip_right = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                knee_left = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
                knee_right = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
                ankle_left = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)
                shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

                knee_angle = calculate_angle_knee(hip_left, knee_left, ankle_left)
                shoulder_tilt = calculate_tilt_shoulder(shoulder_left, shoulder_right)
                knee_distance = calculate_knee_dis(knee_left, knee_right)
                hip_tilt = calculate_tilt_hip(hip_left, hip_right)

                writer.writerow([frame_file, knee_distance, shoulder_tilt, knee_angle, hip_tilt])
                print(f"{frame_file}: Knee Distance = {knee_distance:.2f}, Shoulder Tilt = {shoulder_tilt:.2f}, Knee Angle = {knee_angle:.2f}, Hip Tilt = {hip_tilt:.2f}")

    print("Pose analysis complete. Results saved: squat_analysis.csv")

def control_squat_posture():
    data_dict = {}
    st_failed_front_frame_num = []
    ht_failed_front_frame_num = []
    kd_failed_front_frame_num = []

    with open("squat_analysis.csv", mode="r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            frame_file, knee_dist, shoulder_tilt, knee_angle, hip_tilt = row
            knee_dist = float(knee_dist)
            shoulder_tilt = float(shoulder_tilt)
            knee_angle = float(knee_angle)
            hip_tilt = float(hip_tilt)
            data_dict[frame_file] = (knee_dist, shoulder_tilt, knee_angle, hip_tilt)

    min_knee_angle = min(value[2] for value in data_dict.values())
    min_angle_frame = min(data_dict, key=lambda k: data_dict[k][2])
    print(f"Frame-{min_angle_frame} has min_angle: {min_knee_angle:.2f}")

    for frame_file, (knee_dist, shoulder_tilt, knee_angle, hip_tilt) in data_dict.items():
        if shoulder_tilt < 177:
            st_failed_front_frame_num.append(frame_file)

        if hip_tilt < 177:
            ht_failed_front_frame_num.append(frame_file)

        if knee_dist > 1.2:
            kd_failed_front_frame_num.append(frame_file)

    unique_st_failed_front_frame_num = list(dict.fromkeys(st_failed_front_frame_num))
    unique_ht_failed_front_frame_num = list(dict.fromkeys(ht_failed_front_frame_num))
    unique_kd_failed_front_frame_num = list(dict.fromkeys(kd_failed_front_frame_num))

    return unique_st_failed_front_frame_num, unique_ht_failed_front_frame_num, unique_kd_failed_front_frame_num, min_angle_frame

def show_result(unique_st_failed_front_frame_num, unique_ht_failed_front_frame_num, unique_kd_failed_front_frame_num, min_angle_frame):
    front_frame_files = sorted(os.listdir(front_save_image_dir))
    right_frame_files = sorted(os.listdir(right_save_image_dir))
    normal_delay = 50
    slow_delay = 450

    front_first_frame = cv2.imread(os.path.join(front_save_image_dir, front_frame_files[0]))
    right_first_frame = cv2.imread(os.path.join(right_save_image_dir, right_frame_files[0]))
    front_height, front_width, _ = front_first_frame.shape
    right_height, right_width, _ = right_first_frame.shape
    front_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    right_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    front_out = cv2.VideoWriter(front_result_output_video_path, front_fourcc, 20.0, (front_width, front_height))
    right_out = cv2.VideoWriter(right_result_output_video_path, right_fourcc, 20.0, (right_width, right_height))

    for frame_file in front_frame_files:
        front_frame_path = os.path.join(front_save_image_dir, frame_file)
        right_frame_path = os.path.join(right_save_image_dir, frame_file)
        front_frame = cv2.imread(front_frame_path)
        right_frame = cv2.imread(right_frame_path)
        if front_frame is None or right_frame is None:
            continue

        rgb_front_frame = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
        rgb_right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        front_results = pose.process(rgb_front_frame)
        right_results = pose.process(rgb_right_frame)

        if frame_file in unique_st_failed_front_frame_num or \
           frame_file in unique_ht_failed_front_frame_num or \
           frame_file in unique_kd_failed_front_frame_num or \
           frame_file == min_angle_frame:

            if front_results.pose_landmarks:
                landmarks = front_results.pose_landmarks.landmark
                front_height, front_width, _ = front_frame.shape

                shoulder_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * front_width),
                                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * front_height))
                shoulder_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * front_width),
                                  int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * front_height))
                hip_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * front_width),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * front_height))
                hip_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * front_width),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * front_height))
                knee_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * front_width),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * front_height))
                knee_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * front_width),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * front_height))

                if frame_file in unique_st_failed_front_frame_num:
                    cv2.line(front_frame, shoulder_left, shoulder_right, (0, 0, 255), 3)
                if frame_file in unique_ht_failed_front_frame_num:
                    cv2.line(front_frame, hip_left, hip_right, (0, 0, 255), 3)
                if frame_file in unique_kd_failed_front_frame_num:
                    cv2.line(front_frame, knee_left, knee_right, (0, 0, 255), 3)

            if right_results.pose_landmarks:
                landmarks_right = right_results.pose_landmarks.landmark
                right_height, right_width, _ = right_frame.shape

                hip_right = (int(landmarks_right[mp_pose.PoseLandmark.RIGHT_HIP].x * right_width),
                             int(landmarks_right[mp_pose.PoseLandmark.RIGHT_HIP].y * right_height))
                knee_left = (int(landmarks_right[mp_pose.PoseLandmark.LEFT_KNEE].x * right_width),
                             int(landmarks_right[mp_pose.PoseLandmark.LEFT_KNEE].y * right_height))
                knee_right = (int(landmarks_right[mp_pose.PoseLandmark.RIGHT_KNEE].x * right_width),
                              int(landmarks_right[mp_pose.PoseLandmark.RIGHT_KNEE].y * right_height))
                ankle_right = (int(landmarks_right[mp_pose.PoseLandmark.RIGHT_ANKLE].x * right_width),
                               int(landmarks_right[mp_pose.PoseLandmark.RIGHT_ANKLE].y * right_height))
                if frame_file in unique_st_failed_front_frame_num:
                    cv2.line(right_frame, hip_right, knee_right, (0, 0, 255), 3)
                    cv2.line(right_frame, knee_right, ankle_right, (0, 0, 255), 3)

        if frame_file in unique_st_failed_front_frame_num or \
           frame_file in unique_ht_failed_front_frame_num or \
           frame_file in unique_kd_failed_front_frame_num:
            for _ in range(9):
                front_out.write(front_frame)
                right_out.write(right_frame)
        else:
            front_out.write(front_frame)
            right_out.write(right_frame)

        cv2.imshow('Front View', front_frame)
        cv2.imshow('Right View', right_frame)

        delay = slow_delay if frame_file in unique_st_failed_front_frame_num or \
                            frame_file in unique_ht_failed_front_frame_num or \
                            frame_file in unique_kd_failed_front_frame_num else normal_delay

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

def generate_frames(camera_index=0):
    """Generate frames from the specified camera index."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"🚨 Unable to access camera {camera_index}.")
        return

    while streaming_flags.get(camera_index, False):
        success, frame = cap.read()
        if not success:
            print(f"🚨 Failed to read frame from camera {camera_index}.")
            break

        # Resize frame for consistency
        frame = cv2.resize(frame, (640, 480))

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the correct format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(camera_index=0), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(camera_index=1), mimetype='multipart/x-mixed-replace; boundary=frame') 


@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)

@app.route('/static/videos/<path:filename>')
def serve_static_video(filename):
    # 요청된 파일명을 출력하여 디버깅
    print(f"Serving video file: {filename}")
    # 올바른 디렉토리에서 파일을 제공
    return send_from_directory(os.path.join(app.static_folder, 'videos'), filename)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/squat_ex2')
def squat_ex2():
    return render_template('squat_ex2.html')

@app.route('/squat_guide3')
def squat_guide3():
    return render_template('squat_guide3.html')

@app.route('/squat_check4')
def squat_check4():
    return render_template('squat_check4.html')

@app.route('/setting5', methods=['GET', 'POST'])
def setting5():
    if request.method == 'POST':
        sets = request.form.get('sets')
        reps = request.form.get('reps')
        return redirect(url_for('squat_start6', sets=sets, reps=reps))
    return render_template('setting5.html')

@app.route('/squat_start6', methods=['GET', 'POST'])
def squat_start6():
    if request.method == 'POST':
        sets = request.form.get('sets', 3)
        reps = request.form.get('reps', 10)
    return render_template('squat_start6.html', sets=sets, reps=reps)

@app.route('/squat_end7')
def squat_end7():
    accuracy = 90
    feedback = "무릎을 더 벌리세요."
    return render_template('squat_end7.html', accuracy=accuracy, feedback=feedback)

@app.route('/start_analysis')
def start_analysis():
    recording_pose()
    analyse_pose()
    st_failed, ht_failed, kd_failed, min_angle_frame = control_squat_posture()
    show_result(st_failed, ht_failed, kd_failed, min_angle_frame)
    return redirect(url_for('squat_end7'))

@app.route('/start_stream/<int:camera_index>')
def start_stream(camera_index):
    """Start streaming for the specified camera index."""
    streaming_flags[camera_index] = True
    return '', 204  # Return a no-content response

@app.route('/stop_stream/<int:camera_index>')
def stop_stream(camera_index):
    """Stop streaming for the specified camera index."""
    streaming_flags[camera_index] = False
    return '', 204  # Return a no-content response


# Flask app execution
if __name__ == "__main__":
    print("Flask app running...")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)