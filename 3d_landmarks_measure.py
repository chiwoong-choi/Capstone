import pandas as pd
import numpy as np

# CSV 파일 불러오기
data = pd.read_csv(r'C:\capstone\pose_landmarks.csv', header=0)  

# 데이터프레임을 NumPy 배열로 변환 -> 벡터 계산에 numpy가 적합
data_np = np.array(data, dtype=float)

#프레임 길이 변동 및 측정할 랜드마크 종류 변동 대비 자동 개수 설정
num_frames = data_np.shape[0] #data_np의 열
num_landmarks = data_np.shape[1]//3  #data_np의 행

# (프레임 수, 랜드마크 수, 3D 좌표) 형태로 변환
landmarks_3d = data_np.reshape(num_frames, num_landmarks, 3)

# 변환된 데이터 확인
print(landmarks_3d.shape) #

print(landmarks_3d[0,0])

#프레임별 무릎 각도 확인
angles = []

for frame in range(num_frames):  #landmarks_3d의 0번값= 어느 frame인지
    ldmk_24=landmarks_3d[frame,5] #right hip
    ldmk_26=landmarks_3d[frame,6] #right knee
    ldmk_28=landmarks_3d[frame,7] #right ankle
    
    v1=ldmk_24-ldmk_26
    v2=ldmk_28-ldmk_26
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 > 0 and norm_v2 > 0:  # 0으로 나누는 오류 방지
        dot_product = np.dot(v1, v2)  # 벡터 내적
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)  # 값 범위 조정 (-1 ~ 1)
        angle = np.degrees(np.arccos(cos_theta))  # 라디안 → 도(degree)
    else:
        angle = 0  # 벡터가 0이면 각도 0으로 설정
    
    angles.append(angle)
    
    print(f"Frame {frame + 1}: Angle = {angle:.2f}°") # for문의 frame마다 print