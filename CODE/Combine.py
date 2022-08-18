import cv2
import numpy as np
from keras.models import load_model

model1=load_model('dataset.h5')
model2=load_model('humanmodel1.h5')

video=cv2.VideoCapture(0)

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'Human',1:'cats', 2:'dogs'}
labels_dict2={0:'vivek',1:'abhinav'}

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(50,50))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 50,50, 1))
        result=model1.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        
        print(labels_dict[label])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.imshow("Frame",frame)
    
        if(labels_dict[label]=="Human"):
            resized=cv2.resize(sub_face_img,(255,255))
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 255,255, 1))
            result=model2.predict(reshaped)
            label=np.argmax(result, axis=1)[0]
            print(labels_dict2[label])
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            #cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, labels_dict2[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)  
            #cv2.imshow("Frame",frame)
            
        
        
    k=cv2.waitKey(1)
    if k==ord('s'):
        break

video.release()
cv2.destroyAllWindows()