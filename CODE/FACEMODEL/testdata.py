import cv2
import numpy as np
from keras.models import load_model

model=load_model('dataset.h5')
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict={0:'HUMAN',1:'CAT', 2:'DOG'}

# len(number_of_image), image_height, image_width, channel
img=cv2.imread("kutta.jpg")
frame=cv2.imread("kutta.jpg")
#gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces= faceDetect.detectMultiScale(img, 1.3, 3)
for x,y,w,h in faces:
    sub_face_img=img[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(50,50))
    normalize=resized/255.0
    reshaped=np.reshape(normalize, (1, 50,50, 1))
    result=model.predict(reshaped)
    label=np.argmax(result, axis=1)[0]
    print(label)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
figure = np.concatenate((img, frame), axis=1)   
cv2.imshow("Detected",figure)    
cv2.waitKey(0)
cv2.destroyAllWindows()