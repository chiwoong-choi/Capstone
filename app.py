from flask import Flask, render_template, send_from_directory, abort, Response, url_for
import os
import cv2

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos/<path:filename>')
def serve_video(filename):
    video_dir = os.path.join(app.root_path, "static/videos")  # 절대 경로로 변환
    file_path = os.path.join(video_dir, filename)

    print(f"📂 Checking file path: {file_path}")  # 디버깅을 위한 출력문 추가

    if not os.path.exists(file_path):
        print("🚨 File not found!")  # 파일이 없을 때 터미널에 출력
        abort(404)  # 404 에러 반환

    return send_from_directory(video_dir, filename)

# ✅ 웹캠 실시간 스트리밍을 위한 프레임 생성 함수
def generate_frames(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ✅ 웹캠 스트리밍 라우트 추가
@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(camera_index=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(camera_index=1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squat_ex2')      
def squat_ex2():
    return render_template('squat_ex2.html')

@app.route('/squat_guide3')      
def squat_guide3():
    return render_template('squat_guide3.html')

@app.route('/squat_check4')      
def squat_check4():
    return render_template('squat_check4.html')

@app.route('/setting5')      
def setting5():
    return render_template('setting5.html')

@app.route('/squat_start6')      
def squat_start6():
    return render_template('squat_start6.html')

if __name__ == '__main__':  # ✅ 올바른 실행 조건
    print("🔥 현재 Flask 엔드포인트 목록 🔥", flush=True)
    print(app.url_map, flush=True)  # Flask 엔드포인트 확인용
    app.run(host="0.0.0.0", port=5000, debug=True)
