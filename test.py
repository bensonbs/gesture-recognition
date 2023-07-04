import cv2
import time
import math
import pyautogui
import numpy as np
import mediapipe as mp


from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import warnings
warnings.filterwarnings("ignore")

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def distance_3d(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def mouse2screen(xy):
    xy.x = xy.x*640
    xy.y = xy.y*480
    mouse_center = (640/2,480/2)
    mx = xy.x- mouse_center[0]
    my = xy.y-mouse_center[1]

    screen_center = (1920/2,1080/2)
    sx = int(screen_center[0]+mx*1920/330)
    sy = int(screen_center[1]+my*1080/220)

    return sx,sy



# Variables to store the previous position of the index finger tip
clicks = 0
click_time = 0
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options,num_hands=2)

frame_counter = 0
start_time = time.time()
prev_sx, prev_sy = None, None

while True:
    ret, frame = cap.read()
    if ret:
        # Only convert the frame to RGB once.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb_frame))
        recognizer = vision.GestureRecognizer.create_from_options(options)

        # STEP 4: Recognize gestures in the input image.
        recognition_result = recognizer.recognize(image)

        # STEP 5: Process the result. In this case, visualize it.

        frame_counter += 1


        end_time = time.time()
        fps = frame_counter / (end_time - start_time)

        if len (recognition_result.gestures) > 0:
            top_gesture = recognition_result.gestures[0][0]

            hand_landmarks = recognition_result.hand_landmarks[0]
            gesture = recognition_result.gestures[0][0]
            annotated_image = image.numpy_view()

            # for hand_landmarks,gesture in zip(recognition_result.hand_landmarks,recognition_result.gestures[0]):
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
            image.numpy_view(),
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

            
            if distance_3d(hand_landmarks[4], hand_landmarks[6]) < 0.06:
                pyautogui.doubleClick()

            elif recognition_result.gestures[0][0].category_name == 'Thumb_Up' :
                pyautogui.scroll(100)

            elif recognition_result.gestures[0][0].category_name == 'Thumb_Down' :
                pyautogui.scroll(-100)

            else:
                xy = recognition_result.hand_landmarks[0][5]
                sx, sy = mouse2screen(xy)
                if prev_sx is not None and prev_sy is not None:
                    distance = ((prev_sx - sx) ** 2 + (prev_sy - sy) ** 2) ** 0.5
                    if distance > 10:  # You can set some_threshold as you wish
                        pyautogui.moveTo(sx, sy,duration = 0.0001)
                prev_sx, prev_sy = sx, sy

            text = str(gesture.category_name)
            cv2.putText(annotated_image, str(round(distance_3d(hand_landmarks[4], hand_landmarks[6]),2)), (10, 80), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 255), 1, cv2.LINE_AA)
            bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', bgr_annotated_image)

        else:
            cv2.imshow('frame', frame)

             # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()