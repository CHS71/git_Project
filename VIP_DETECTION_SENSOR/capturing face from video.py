#동영상에서 얼굴잘라내는 코드

# !/opt/local/bin/python
# -*- coding: utf-8 -*-
import cv2

## 재생할 파일

VIDEO_FILE_PATH = 'C:\\Users\\CHS\\Desktop\\oracle U\\test3.t.mp4'

cap = cv2.VideoCapture(VIDEO_FILE_PATH)

if (cap.isOpened() == False):
    print("Unable to read camera feed")
    exit()
titles = ['test']
for t in titles:
    cv2.namedWindow(t)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print('width {0}, height {1}, fps {2}'.format(width, height, fps))

face_cascade = cv2.CascadeClassifier()
face_cascade.load('c:\\video\\haarcascade_frontalface_default.xml')

imgNum = 0
while (True):
    ret, frame = cap.read()
    if frame is None:
        break;

    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayframe, (5, 5), 0)
    faces = face_cascade.detectMultiScale(blur, 1.8, 2, 0, (50, 50))

    for (x, y, w, h) in faces:
        if imgNum % 4 == 0:
            cropped = frame[y:y + h, x:x + w]
            cv2.imwrite("C:\\Users\\CHS\\Desktop\\oracle U\\face\\facess" + str(imgNum) + ".png", cropped)
        imgNum += 1
    cv2.imshow(titles[0], frame)

    if cv2.waitKey(1) == 27:
        break;

cap.release()

cv2.destroyAllWindows()