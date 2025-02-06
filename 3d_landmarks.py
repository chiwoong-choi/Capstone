import numpy as np
import pyvista as pv
import pandas as pd
# 주어진 데이터 (33개 랜드마크의 x, y, z 좌표)
data = pd.read_csv(r'C:\capstone\1pixel.csv')

# 3D 좌표로 나누기
landmarks_3d = np.array(data).reshape(-1, 3)

# PyVista에서 사용할 수 있는 포인트 클라우드 생성
point_cloud = pv.PolyData(landmarks_3d)

# Plotter 생성
plotter = pv.Plotter()

# 포인트 클라우드 시각화
plotter.add_points(point_cloud, color='red', point_size=10)

# POSE_CONNECTIONS에 따른 연결선 추가 (기본 포즈 연결선)
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), 
    (5, 6), (6, 7), (7, 8), (8, 9)]

# 연결선 그리기
for connection in POSE_CONNECTIONS:
    start = landmarks_3d[connection[0]]
    end = landmarks_3d[connection[1]]
    plotter.add_lines(np.array([start, end]), color='blue')

# 3D 시각화
plotter.show()
