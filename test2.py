import mediapipe as mp
import cv2
mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

pose=mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

image_path= 'd.jpg'
image=cv2.imread(image_path)

image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

results=pose.process(image_rgb)

if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
else:
    print("asd")
    
cv2.imwrite('output.jpg',image)
cv2.imshow('Pose',image)
cv2.waitKey(0)
cv2.destroyAllWindows()