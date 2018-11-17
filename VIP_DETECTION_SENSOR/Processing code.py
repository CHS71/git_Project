#Processing code





### 특정 경로에 있는 이미지들을 한장씩 열어서 얼굴이 있는지 확인하고 얼굴이 있다면 얼굴부분만 잘라서 다른 폴더에 저장하는 소스.


import cv2
import glob

face_cascade = cv2.CascadeClassifier('c:\\video\\haarcascade_frontalface_default.xml')

imglist = glob.glob('c:\\googleimg\\*.jpg')
## 특정 경로에 있는 .jpg 파일을 모두 imglist 변수에 담는다.
imgNum = 0
for img_file in imglist:

    img = cv2.imread(img_file)
    # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        # 이미지를 저장
        cv2.imwrite("c:\\video\\face1\\face" + str(imgNum) + ".png", cropped)
        imgNum += 1

print('완료')







##  비정상적인 파일을 모두 삭제해주는 소스


from os.path import getsize
from os import remove
from PIL import Image
import glob
import sys

path = 'C:\\video\\face\\'
imglist = glob.glob(path + '*.png')

cnt = 0
cnt2 = 0
for img_file in imglist:
    if not getsize(img_file) or getsize(img_file) <= 2000:  # 용량이 0일때 삭제
        cnt += 1
        remove(img_file)

    elif getsize(img_file):  # 가로/세로 또는 세로/가로 비율이 비정상 적일 경우 삭제
        img = Image.open(img_file)
        img_h, img_w = img.size[0], img.size[1]
        if img_h / img_w >= 3 or img_w / img_h >= 3:
            cnt2 += 1
            img.close()
            remove(img_file)


print('크기가 비정상인 파일 : ' + str(cnt) + '건 삭제')
print('비정상 비율인 파일 : ' + str(cnt2) + '건 삭제')












## 특정 폴더에 존재하는 이미지 파일을
# 1. 비정상 크기, 비율 파일 삭제
# 2. 이미지 크로핑
# 3. 사이즈 조절
# 하는 코드입니다..
#####data_preprocessing.py


import face_recognition
import glob
import cv2

img_size = 64
img_path = './img_data/'

#####

from os.path import getsize
from os import remove
from PIL import Image
import sys

print('[INFO] 크기 및 비율이 비정상인 데이터를 삭제합니다.')

imglist = glob.glob(img_path + '*.png')
cnt = 0
cnt2 = 0
for img_file in imglist:
    if not getsize(img_file) or getsize(img_file) <= 2000:  # 용량이 0일때 삭제
        cnt += 1
        remove(img_file)

    elif getsize(img_file):  # 가로/세로 또는 세로/가로 비율이 비정상 적일 경우 삭제
        img = Image.open(img_file)
        img_h, img_w = img.size[0], img.size[1]
        if img_h / img_w >= 3 or img_w / img_h >= 3:
            cnt2 += 1
            img.close()
            remove(img_file)

print('----크기가 비정상인 파일 : ' + str(cnt) + '건 삭제')
print('----비정상 비율인 파일 : ' + str(cnt2) + '건 삭제', end='\n\n')

####

print('[INFO]', img_path, '의 이미지에서 얼굴을 추출합니다.')
print('----이미지에서 얼굴을 추출 중...')
imglist = glob.glob(img_path + '*.jpg')
imglist2 = glob.glob(img_path + '*.png')
imglist.extend(imglist2)

imgNum = 0
cnt5 = 0
for img_file in imglist:
    cnt5 += 1
    print('{}\r'.format(str(cnt5) + '/' + str(len(imglist)) + '진행 중..'), end='')
    sys.stdout.flush()
    print(cnt5)
    img = cv2.imread(img_file)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb, model='cnn')

    for top, right, bottom, left in faces:
        cropped = img[top: bottom, left: right]
        # 이미지를 저장
        cv2.imwrite(img_path + 'cvt_' + str(imgNum) + ".png", cropped)
        imgNum += 1
    remove(img_file)

print('----' + str(imgNum) + '건의 얼굴을 추출했습니다. ', end='\n\n')
###img resize

print('[INFO]', '이미지의 사이즈를', img_size, '으로 수정합니다.')

imglist = glob.glob(img_path + "*png")

img = Image.open(imglist[0])

for img_path in imglist:
    img = Image.open(img_path)
    img.resize((img_size, img_size)).save(img_path)

print('[INFO] 작업이 성공적으로 수행되었습니다.')














### 사이즈 통일


# import glob
# from PIL import Image
#
#
#
# imglist =glob.glob("C:\\video\\oracleU\\*jpeg")
#
# # img = Image.open(imglist[0])
#
# for img_path in imglist:
#     img = Image.open(img_path)
#     img.resize((64,64)).save(img_path)
#
#



















#####웹캠을 이용해서 실시간으로 사람얼굴을 인식하고 인식한 이미지를 특정 경로에 저장하는 소스입니다. 기존의 소스처럼 동영상을 따로 저장하지 않고, 영상을 확인 할 때는 초록색 네모가 보이지만 저장된 이미지는 네모가 잡히지 않습니다.

#
# import cv2
#
# #!/opt/local/bin/python
# # -*- coding: utf-8 -*-
# import cv2
#
# #재생할 파일
# #VIDEO_FILE_PATH = 'c:\\data\\video\\test3.mp4'
#
# # 동영상 파일 열기
# #cap = cv2.VideoCapture(VIDEO_FILE_PATH)
# cap = cv2.VideoCapture(0)
# #잘 열렸는지 확인
# # if cap.isOpened() == False:
# #     print ('Can\'t open the video (%d)' % (VIDEO_FILE_PATH))
# #     exit()
# if (cap.isOpened() == False):
#     print("Unable to read camera feed")
#     exit()
# titles = ['test']
# #윈도우 생성 및 사이즈 변경
# for t in titles:
#     cv2.namedWindow(t)
#
# #재생할 파일의 넓이 얻기
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# #재생할 파일의 높이 얻기
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# #재생할 파일의 프레임 레이트 얻기
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# print('width {0}, height {1}, fps {2}'.format(width, height, fps))
#
# #XVID가 제일 낫다고 함.
# #linux 계열 DIVX, XVID, MJPG, X264, WMV1, WMV2.
# #windows 계열 DIVX
# #저장할 비디오 코덱
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# #저장할 파일 이름
# filename = 'c:\\data\\video\\sprite_with_face_detect.avi'
#
# #파일 stream 생성
# #out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
# #filename : 파일 이름
# #fourcc : 코덱
# #fps : 초당 프레임 수
# #width : 넓이
# #height : 높이
#
# #얼굴 인식용
# face_cascade = cv2.CascadeClassifier()
# face_cascade.load('c:\\data\\video\\haarcascade_frontalface_default.xml')
#
# imgNum=0
# while(True):
#     #파일로 부터 이미지 얻기
#     ret, frame = cap.read()
#     #더 이상 이미지가 없으면 종료
#     #재생 다 됨
#     if frame is None:
#         break;
#
#     #얼굴인식 영상 처리
#     grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur =  cv2.GaussianBlur(grayframe,(5,5), 0)
#     faces = face_cascade.detectMultiScale(blur, 1.8, 2, 0, (50, 50))
#     imgNum+=1
#     #원본 이미지에 얼굴 인식된 부분 표시
#     for (x,y,w,h) in faces:
# #         cx = int(x+(w/2))
# #         cy = int(y+(h/2))
# #         cr = int(w/2)
#         #cv2.##(frame,(cx,cy),cr,(0,255,0),3)
#         cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),3)
#         cropped = frame[y :y + h , x :x + w ]
#         cv2.imwrite("c:\\data\\video\\face\\face" + str(imgNum) + ".png", cropped)
#     # 얼굴 인식된 이미지 화면 표시0
#     cv2.imshow(titles[0],frame)
#
#     # 인식된 이미지 파일로 저장
#     #out.write(frame)
#
#     #1ms 동안 키입력 대기
#     if cv2.waitKey(1) == 27:
#         break;
#
#
# #재생 파일 종료
# cap.release()
# #저장 파일 종료
# #out.release()
# #윈도우 종료
# cv2.destroyAllWindows()






import cv2
# !/opt/local/bin/python
# -*- coding: utf-8 -*-
# import cv2
#
# # 재생할 파일
# VIDEO_FILE_PATH = 'c:\\data\\video\\test3.mp4'
#
# cap = cv2.VideoCapture(VIDEO_FILE_PATH)
#
# if (cap.isOpened() == False):
#     print("Unable to read camera feed")
#     exit()
# titles = ['test']
# for t in titles:
#     cv2.namedWindow(t)
#
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# print('width {0}, height {1}, fps {2}'.format(width, height, fps))
#
# face_cascade = cv2.CascadeClassifier()
# face_cascade.load('c:\\data\\video\\haarcascade_frontalface_default.xml')
#
# imgNum = 0
# while (True):
#     ret, frame = cap.read()
#     if frame is None:
#         break;
#
#     grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grayframe, (5, 5), 0)
#     faces = face_cascade.detectMultiScale(blur, 1.8, 2, 0, (50, 50))
#
#     for (x, y, w, h) in faces:
#         if imgNum % 4 == 0:
#             cropped = frame[y:y + h, x:x + w]
#             cv2.imwrite("c:\\data\\video\\face\\face" + str(imgNum) + ".png", cropped)
#         imgNum += 1
#     cv2.imshow(titles[0], frame)
#
#     if cv2.waitKey(1) == 27:
#         break;
#
# cap.release()
#
# cv2.destroyAllWindows()








