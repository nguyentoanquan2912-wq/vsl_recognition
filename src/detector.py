import cv2
import mediapipe as mp
import numpy as np

def extract_landmarks_from_image(image_path):
    """
    Trích xuất tọa độ 21 điểm của bàn tay từ hình ảnh đầu vào sử dụng MediaPipe.
    Hữu ích cho các kịch bản offline / batch processing.
    """
    # Khởi tạo mediapipe hands
    mp_hands = mp.solutions.hands
    
    # Sử dụng với context manager
    with mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        
        # Đọc hình ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc ảnh: {image_path}")
            return None
            
        # Chuyển đổi BGR (OpenCV) sang RGB (MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Xử lý hình ảnh
        results = hands.process(image_rgb)
        
        # Nếu phát hiện được bàn tay
        if results.multi_hand_landmarks:
            # Lấy bàn tay đầu tiên (vì max_num_hands=1)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Trích xuất x, y, z
            landmarks_array = []
            for landmark in hand_landmarks.landmark:
                landmarks_array.append([landmark.x, landmark.y, landmark.z])
                
            return np.array(landmarks_array)
            
    # Không tìm thấy bàn tay
    return None
