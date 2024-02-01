import cv2

def print_camera_indices():
    for i in range(10):  # Check indices from 0 to 9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            cap.release()

print_camera_indices()
