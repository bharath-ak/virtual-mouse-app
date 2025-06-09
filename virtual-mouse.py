import cv2
import mediapipe as mp
import numpy as np
import random
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

mpHands = mp.solutions.hands
draw = mp.solutions.drawing_utils

def detect_mode(landmark_list):
    def is_finger_up(lm_base, lm_tip):
        return lm_tip[1] < lm_base[1]

    if len(landmark_list) < 21:
        return None

    finger_states = {
        'thumb':  is_finger_up(landmark_list[2], landmark_list[4]),
        'index':  is_finger_up(landmark_list[6], landmark_list[8]),
        'middle': is_finger_up(landmark_list[10], landmark_list[12]),
        'ring':   is_finger_up(landmark_list[14], landmark_list[16]),
        'pinky':  is_finger_up(landmark_list[18], landmark_list[20]),
    }

    fingers_up = [finger for finger, up in finger_states.items() if up]
    up_count = len(fingers_up)

    if 'ring' in fingers_up and 'pinky' in fingers_up and up_count >= 2:
        if 'middle' in fingers_up:
            return 'mouselc'
        elif 'index' in fingers_up:
            return 'mouserc'
        else:
            return 'mouse'
    elif 'index' in fingers_up and 'middle' in fingers_up and 'ring' in fingers_up and up_count >= 4:
        return 'volume'
    elif 'index' in fingers_up and 'middle' in fingers_up and 'pinky' in fingers_up and up_count >= 4:
        return 'brightness'
    elif up_count == 1:
        return 'screenshot'
    return None

def get_angle(x, y, z):
    radian = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
    angle = np.abs(np.degrees(radian))
    return angle

def get_dist(landmark_list):
    if len(landmark_list) < 2:
        return
    (x1,y1),(x2,y2) = landmark_list[0],landmark_list[1]
    euc_dist = np.hypot(x2-x1,y2-y1)
    return np.interp(euc_dist,[0,1],[0,100])

def get_finger_tips(hand_landmarks):
    return {
        'thumb':  hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP],
        'index':  hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP],
        'middle': hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP],
        'ring':   hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP],
        'pinky':  hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP],
    }

def left_click(landmark_list):
    return(get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) < 50 and
           get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) > 90 and
           get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90 and
           get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) > 90)

def right_click(landmark_list):
    return(get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 50 and
           get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) > 90 and 
           get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90 and
           get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) > 90)

def double_click(landmark_list):
    return(get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) < 50 and
           get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 50 and
           get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90 and
           get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) > 90) 

def screen_shot(landmark_list):
    return (get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) < 50 and
            get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) < 50)

def get_volume(landmark_list):
    return (get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90 and
            get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) < 50)

def get_brightness(landmark_list):
    return (get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) < 50 and
            get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) > 90) 

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mpHands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        mirror_img = cv2.flip(img, 1)
        frameRGB = cv2.cvtColor(mirror_img, cv2.COLOR_BGR2RGB)
        processed = self.hands.process(frameRGB)         
        landmark_list = []        

        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0] 
            draw.draw_landmarks(mirror_img, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.append((lm.x, lm.y))

            h, w, _ = mirror_img.shape

            current_mode = detect_mode(landmark_list)
            if len(landmark_list) >= 21:
                finger_tips = get_finger_tips(hand_landmarks)
                thumb_index_dist = get_dist([landmark_list[4],landmark_list[8]])
                index_tip = finger_tips['index']
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(mirror_img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(mirror_img, f"Pointer: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if current_mode == 'mouselc' and left_click(landmark_list):
                    cv2.putText(mirror_img, "LEFT CLICK", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif current_mode == 'mouserc' and right_click(landmark_list):
                    cv2.putText(mirror_img, "RIGHT CLICK", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif current_mode == 'mouse' and double_click(landmark_list):
                    cv2.putText(mirror_img, "DOUBLE CLICK", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif current_mode == 'screenshot' and screen_shot(landmark_list):
                    filename = f"screenshot_{random.randint(1000, 9999)}.png"
                    cv2.imwrite(filename, mirror_img)
                    cv2.putText(mirror_img, "SCREENSHOT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif current_mode == 'volume' and get_volume(landmark_list):
                    thumb_index_dist = np.clip(thumb_index_dist, 25, 50)
                    volume_level = np.interp(thumb_index_dist, [25, 50], [0.0, 1.0])
                    volume_level = np.clip(volume_level, 0.0, 1.0)
                    # Draw volume bar
                    bar_x, bar_y = 50, 100
                    bar_height = 300
                    bar_width = 30
                    filled = max(0, int(bar_height * volume_level))
                    # Outline
                    cv2.rectangle(mirror_img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                    # Fill
                    cv2.rectangle(mirror_img,(bar_x, bar_y + bar_height - filled), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
                    # Display percentage
                    cv2.putText(mirror_img, f"{int(volume_level * 100)}%", (bar_x - 10, bar_y + bar_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # Label
                    cv2.putText(mirror_img, "Volume", (bar_x - 10, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif current_mode == 'brightness' and get_brightness(landmark_list):
                    thumb_index_dist = np.clip(thumb_index_dist, 25, 50)
                    brightness_level = np.interp(thumb_index_dist, [25, 50], [0.0, 1.0])
                    brightness_level = np.clip(brightness_level, 0.0, 1.0)
                    bar_x, bar_y = 50, 100
                    bar_height = 300
                    bar_width = 30
                    filled = max(0, int(bar_height * brightness_level))
                    # Outline
                    cv2.rectangle(mirror_img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                    # Fill
                    cv2.rectangle(mirror_img,(bar_x, bar_y + bar_height - filled), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
                    # Display percentage
                    cv2.putText(mirror_img, f"{int(brightness_level * 100)}%", (bar_x - 10, bar_y + bar_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # Label
                    cv2.putText(mirror_img, "Brightness", (bar_x - 10, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(mirror_img, format="bgr24")

rtc_config = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    }
)

webrtc_streamer(
    key="virtual-mouse",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
