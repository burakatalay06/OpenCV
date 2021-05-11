import cv2
import imageio


smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')


def detect(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)             #1.3 oranında resmi küçültür ve hata payı muhabbeti 5 te #
    for(x,y,w,h) in faces:                                          #face üst satırda bulundu bu sefer bulunan her face for döngüsüne sokuluyor#
        cv2.rectangle(frame,(x, y), (x+w, y+h), (255,0,0),3)        # dikdörtgen çizdirmek için gereken komut...#
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        for(ex, ey, ew, eh) in smile:
            cv2.rectangle(color_face,(ex, ey),(ex+ew, ey+eh), (0,255,0),2)  #gülücüğe dikdörtgen çizdi#
    return frame

reader = imageio.get_reader('1.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4',fps=fps )

for i,frame in enumerate(reader):
    frame = detect(frame)
    writer.append_data(frame)   #bulunan yüz ve gülmeleri tek tek output.mp4 e ekliyor#

    print(i)

writer.close()




