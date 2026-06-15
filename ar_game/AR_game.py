import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import math
import time
import random
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode

from PIL import Image as PILImage
import sys

# global constants and variables

VIDEO_ID = 0
TARGET_COUNT = 5

board = None
board_matrix = None
last_board = None
last_board_matrix = None
finger_pos = None

lost_frames = 0
MAX_LOST_FRAMES = 5


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


if len(sys.argv) > 1:
    VIDEO_ID = int(sys.argv[1])

# Camera setup
cap = cv2.VideoCapture(VIDEO_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, frame = cap.read()

if not ret or frame is None:
    raise RuntimeError("camera could not start")

CAM_HEIGHT, CAM_WIDTH = frame.shape[:2]

# score variable and label
score = 0

score_label = pyglet.text.Label(
    "",
    x=20, y=CAM_HEIGHT-30,
    color=(255,255,255,255),
    font_size=24
)

# Aruco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# window
window = pyglet.window.Window(CAM_WIDTH, CAM_HEIGHT, "game")

# mediapipe hand setup
base_options = python.BaseOptions(model_asset_path = str(MODEL_PATH))
options = hand_landmarker.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = RunningMode.VIDEO,
    num_hands = 1,
    min_hand_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
landmarker = hand_landmarker.HandLandmarker.create_from_options(options)

SMOOTH_ALPHA = 0.6
smoothed_point = None

# game objects

class Target:
    def __init__(self, width, height):
        self.radius = 30
        self.x = random.randint(self.radius, width-self.radius)
        self.y = random.randint(self.radius, height-self.radius)

targets = [Target(CAM_WIDTH, CAM_HEIGHT) for _ in range(TARGET_COUNT)]

# converts OpenCV image to PIL image and then to pyglet texture
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = PILImage.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

# sort points in space
def sort_points(pts):
    pts = np.array(pts)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.float32([top_left, top_right, bottom_right, bottom_left])

# extract game window
def get_board(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    # get marker positions
    marker_points = {}

    for corner, marker_id in zip(corners, ids.flatten()):
        center = corner[0].mean(axis=0)
        marker_points[int(marker_id)] = center

    # check ids
    required = [0,1,2,3]
    
    if not all(i in marker_points for i in required):
        return None
    
    # select source points
    src = sort_points([
        marker_points[0],
        marker_points[1],
        marker_points[2],
        marker_points[3]
    ])

    # select destination points
    dst = np.float32([
        [0,0],
        [CAM_WIDTH,0],
        [CAM_WIDTH,CAM_HEIGHT],
        [0,CAM_HEIGHT]
    ])

    # transform
    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(frame, matrix, (CAM_WIDTH, CAM_HEIGHT))

    return warped, matrix

# finge tracking
def detect_finger(board):
    global smoothed_point

    if board is None:
        return None
    
    # mp requires RGB
    rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format = ImageFormat.SRGB, data = rgb)
    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        smoothed_point = None
        return None
    
    # index fingertip is landmark 8 in mediapipe
    lm = result.hand_landmarks[0][8]
    x_px = int(lm.x * board.shape[1])
    y_px = int(lm.y * board.shape[0])

    # smoothing
    if smoothed_point is None:
        smoothed_point = np.array([x_px, y_px], dtype=float)
    else:
        smoothed_point = SMOOTH_ALPHA * np.array([x_px, y_px]) + (1 - SMOOTH_ALPHA) * smoothed_point

    return (int(smoothed_point[0]), int(smoothed_point[1]))

# collision detection
def check_collisions(finger_pos):

    global score, targets

    if finger_pos is None:
        return
    
    fx, fy = finger_pos

    for target in targets[:]:
        dist = math.sqrt(
            (fx - target.x) ** 2 +
            (fy - target.y) ** 2
        )

        if dist < target.radius:
            targets.remove(target)

            targets.append(Target(CAM_WIDTH, CAM_HEIGHT))

            score += 1

# draw game window
def draw_game(board, finger_pos):

    for target in targets:
        cv2.circle(board, (target.x, target.y), target.radius,
                   (0, 255, 0), -1)
        
    if finger_pos is not None:
        cv2.circle(board, finger_pos, 15, (0, 0, 255), -1)
    
# Render game state
@window.event

def on_draw():
    window.clear()

    if board is None:
        if frame is not None:
            display = cv2.flip(frame, 1)
            img = cv2glet(display, "BGR")
            img.blit(0, 0)
        return
    
    draw_game(board, finger_pos)
    display = cv2.flip(board, 1)
    img = cv2glet(display, "BGR")
    img.blit(0, 0)

    score_label.text = f"Score: {score}"
    score_label.draw()

# update game state 

def update(dt):
    global frame, board, board_matrix, last_board, last_board_matrix, finger_pos
    global MAX_LOST_FRAMES, lost_frames
    
    ret, frame = cap.read()
    if not ret:
        return 
    
    new_board = get_board(frame)

    if new_board is not None:
        board, board_matrix = new_board
        last_board = board.copy()
        last_board_matrix = board_matrix
        lost_frames = 0
    else:
        lost_frames += 1
        if lost_frames <= MAX_LOST_FRAMES and last_board is not None:
            board = last_board.copy()
            board_matrix = last_board_matrix
        else:
            board = None
            board_matrix = None
            finger_pos = None
            return
        
    # detect finger in whole camera frame
    full_finger = detect_finger(frame)
    finger_pos = None

    if full_finger is not None and board_matrix is not None:
        point = np.array([[full_finger]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, board_matrix)
        finger_pos = tuple(map(int, mapped[0, 0]))

    if finger_pos is not None:
        check_collisions(finger_pos)

pyglet.clock.schedule_interval(update, 1/60)

# run game
if __name__ == "__main__":
    pyglet.app.run()

landmarker.close()
cap.release()
