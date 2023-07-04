import cv2
import math
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from pynput.mouse import Button, Controller

# 計算 3D 空間中兩點的距離
def distance_3d(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

# 將手部座標轉換為螢幕座標
def mouse2screen(xy):
    mouse_center = np.array([320, 240])
    screen_center = np.array([960, 540])
    screen_scale = np.array([pixel_x / 330, pixel_y / 220])

    m = (np.array([xy.x, xy.y]) * [640, 480] - mouse_center)
    s = screen_center + m * screen_scale

    return tuple(map(int, s))

# 設定攝影機捕捉設定
def setup_capture():
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# 設定手勢辨識器
def setup_recognizer():
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=1)
    return vision.GestureRecognizer.create_from_options(options)

# 處理每一個影格
def process_frame(recognizer, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb_frame))
    recognition_result = recognizer.recognize(image)

    if len(recognition_result.hand_landmarks) > 0:
        hand_landmarks = recognition_result.hand_landmarks[0]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp_drawing.draw_landmarks(
            image.numpy_view(),
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        action = get_gesture_action(hand_landmarks, recognition_result.gestures[0][0].category_name)
        if action:
            action()

        text = str(recognition_result.gestures[0][0].category_name)
        annotated_image = image.numpy_view()
        cv2.putText(annotated_image, text, 
                    (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
        
        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
    return frame

def press():
    global hold
    hold = True
    mouse.press(Button.left)

def release():
    global hold
    hold = False
    mouse.release(Button.left)

def get_gesture_action(hand_landmarks, gesture):
    actions = {
        'Pointing_Up': lambda:  mouse.scroll(0, 1),  # 捲動滑鼠向上
        'Victory': lambda:  mouse.scroll(0, -1),  # 捲動滑鼠向下
        'Open_Palm': lambda: move_mouse(hand_landmarks)  # 移動滑鼠
    }
    
    if gesture == 'Open_Palm':
        # 檢查兩個關鍵點之間的距離是否小於 0.07
        if distance_3d(hand_landmarks[4], hand_landmarks[6]) < 0.07:
            if not hold:
                return lambda: press()  # 若未按住，觸發按下函數
        
        # 檢查是否已按住且距離是否大於 0.07
        if hold and (distance_3d(hand_landmarks[4], hand_landmarks[6]) > 0.07):
            return lambda: release()  # 若已按住，觸發放開函數

    # 根據給定的手勢返回對應的動作
    return actions.get(gesture, None)


def move_mouse(hand_landmarks):
    # 獲取特定關鍵點的 x 和 y 座標
    xy = hand_landmarks[5]
    sx, sy = mouse2screen(xy)  # 將座標轉換為螢幕座標
    prev_sx, prev_sy = mouse.position  # 獲取上一個滑鼠位置
    
    # 計算前一個位置和當前位置之間的距離
    distance = ((prev_sx - sx) ** 2 + (prev_sy - sy) ** 2) ** 0.5
    
    if distance > 15:
        mouse.position = (sx, sy)  # 將滑鼠移動到新位置

def main():
    cap = setup_capture()  # 設置視訊捕獲
    recognizer = setup_recognizer()  # 設置手勢識別器

    while True:
        ret, frame = cap.read()  # 從視訊捕獲中讀取一幀
        
        if ret:
            frame = process_frame(recognizer, frame)  # 處理幀以識別手勢
            cv2.imshow('frame', frame)  # 顯示處理後的幀
        
        # 如果按下 'q' 鍵，則跳出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 釋放視訊捕獲
    cv2.destroyAllWindows()  # 

if __name__ == '__main__':
    pixel_x, pixel_y = 1920, 1080
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mouse = Controller()
    hold = False
    main()

