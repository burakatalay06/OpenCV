import cv2
import imageio



smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')


def detect(frame):
    smile_number = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)             #1.3 oranında resmi küçültür ve hata payı muhabbeti 5 te #
    for(x,y,w,h) in faces:                                          #face üst satırda bulundu bu sefer bulunan her face for döngüsüne sokuluyor#
        cv2.rectangle(frame,(x, y), (x+w, y+h), (0,0,255),3)        # dikdörtgen çizdirmek için gereken komut...#
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        for(ex, ey, ew, eh) in smile:
            cv2.rectangle(color_face,(ex, ey),(ex+ew, ey+eh), (0,255,25),2)  #gülücüğe dikdörtgen çizdi#
            smile_number = 1
    if smile_number == 1:
        print("ne güzel gülüyorsun :D")

    else:
        print("gülsene lan")

    return frame




video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    canvas = detect(frame)

    cv2.imshow('Video', canvas)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


