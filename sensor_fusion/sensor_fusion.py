import cv2
import cv2.aruco as aruco
import time
import sys
import pyglet
import numpy as np
from DIPPID import SensorUDP

# camera id
VIDEO_ID = 0

# use UDP
PORT = 5700
sensor = SensorUDP(PORT)

# global state variables
camera_pos = None
pred_pos = np.array([0.5, 0.5], dtype=np.float32)
velocity = np.zeros(2, dtype=np.float32)
alpha = 0.85
acc_scale = 2.0
last_time = time.time()
reset_flag = False
acc_x, acc_y = 0, 0
grav_x, grav_y = 0, 0
current_frame = None

if len(sys.argv) > 1:
    VIDEO_ID = int(sys.argv[1])

# Camera setup
cap = cv2.VideoCapture(VIDEO_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, frame = cap.read()

if not ret or frame is None:
    raise RuntimeError("camera could not start")

CAM_HEIGHT, CAM_WIDTH = frame.shape[:2]

# text label
text_label = pyglet.text.Label("", x = CAM_WIDTH / 2, y = CAM_HEIGHT - 20, 
                               color = (255, 255, 255, 255))

# Aruco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# pyglet window
window = pyglet.window.Window(CAM_WIDTH, CAM_HEIGHT, "game")

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

    return warped 

# function to get aruco marker position
def find_marker_position(frame, marker_id):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None
    
    for corner, detected_id in zip(corners, ids.flatten()):
        if int(detected_id) == marker_id:
            return corner[0].mean(axis = 0)
    return None

# define how to handle acceleration data
def handle_acceleration(data):
    global acc_x, acc_y

    if (sensor.has_capability('accelerometer')):

        acc_x = sensor.get_value('accelerometer')['x']
        acc_y = sensor.get_value('accelerometer')['y']

sensor.register_callback('accelerometer', handle_acceleration)

# define how to handle gravity data
def handle_gravity(data):
    global grav_x, grav_y

    if (sensor.has_capability('gravity')):

        grav_x = sensor.get_value('gravity')['x'] / 9.81
        grav_y = sensor.get_value('gravity')['y'] / 9.81

sensor.register_callback('gravity', handle_gravity)

# define how to handle button_1 press
def handle_button(data):
    global reset_flag

    if int(data) == 1:

        reset_flag = True

sensor.register_callback('button_1', handle_button)

@window.event
def on_key_press(key, _):
    global alpha
    if key == pyglet.window.key.LEFT:
        alpha = max(0.0, alpha - 0.05)
    if key == pyglet.window.key.RIGHT:
        alpha = max(0.0, alpha + 0.05)

@window.event
def on_draw():
    window.clear()

    if current_frame is not None:
        rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        image = pyglet.image.ImageData(CAM_WIDTH, CAM_HEIGHT, 'RGB', rgb[::-1, :, :].tobytes(),
                                       pitch = CAM_WIDTH * 3)
        image.blit(0, 0)

    if camera_pos is not None:
        red_dot = pyglet.shapes.Circle(
            camera_pos[0] * CAM_WIDTH,
            (1.0 - camera_pos[1]) * CAM_HEIGHT, 10,
            color = (255, 0, 0))
        red_dot.draw()
        
    green_dot = pyglet.shapes.Circle(
        pred_pos[0] * CAM_WIDTH,
        (1.0 - pred_pos[1]) * CAM_HEIGHT, 10,
        color = (0, 255, 0))
    
    green_dot.draw()
    text_label.draw()


# update function
def update(dt):
    global camera_pos, pred_pos, velocity, last_time
    global reset_flag, text_label, current_frame

    ret, frame = cap.read()
    if not ret or frame is None:
        return
    
    board_frame = get_board(frame)

    if board_frame is not None:
        current_frame = board_frame
        marker_center = find_marker_position(board_frame, marker_id = 5)
    else:
        current_frame = frame
        marker_center = None
    
    if marker_center is not None:
        camera_pos = np.array([marker_center[0] / CAM_WIDTH, marker_center[1] / CAM_HEIGHT,], dtype=np.float32)
        camera_pos = np.clip(camera_pos, 0.0, 1.0)
        if reset_flag:
                pred_pos = camera_pos.copy()
                velocity = np.zeros(2, dtype = np.float32)
                reset_flag = False
    
    if reset_flag:
        reset_flag = False
        if camera_pos is not None:
            pred_pos = camera_pos.copy()
            velocity = np.zeros(2, dtype = np.float32)

    dt = time.time() - last_time
    last_time = time.time()

    lin_x = acc_x - grav_x
    lin_y = acc_y - grav_y

    predicted_x = pred_pos[0] + velocity[0] * dt + lin_x * acc_scale * dt * dt
    predicted_y = pred_pos[1] + velocity[1] * dt + lin_y * acc_scale * dt * dt

    velocity[0] += lin_x * acc_scale * dt
    velocity[1] += lin_y * acc_scale * dt

    predicted = np.array([predicted_x, predicted_y], dtype = np.float32)

    if camera_pos is not None:
        pred_pos = alpha * predicted + (1.0 - alpha) * camera_pos
    else:
        pred_pos = predicted
    
    pred_pos = np.clip(pred_pos, 0.0, 1.0)

    text_label.text = (
        f"alpha={alpha:.2f}")
    
pyglet.clock.schedule_interval(update, 1/30)
pyglet.app.run()

cap.release()
sensor.disconnect()
    
