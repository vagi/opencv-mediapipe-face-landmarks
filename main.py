import cv2
import mediapipe as mp


print(cv2.__version__)

width = 1280
height = 720

cam = cv2.VideoCapture(1)    # in case I'm using Terminal my built-in camera=3, in case of PyCharm camera=1
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

my_face_mesh = mp.solutions.face_mesh.FaceMesh(3, 1)
mp_draw = mp.solutions.drawing_utils

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_color = (0, 255, 255)
font_thickness = 1

while cam.isOpened():
    ignore, frame = cam.read()
    if ignore:
        frame = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = my_face_mesh.process(frame_rgb)
        # print(results.multi_face_landmarks)

        if results.multi_face_landmarks is not None:
            for face_landmark in results.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, face_landmark)

        cv2.imshow('my WEBcam', frame)
        cv2.moveWindow('my WEBcam', 0, 0)
        if cv2.waitKey(1) & 0xff ==ord('q'):
            break

cam.release()
