import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

model_path = 'mymodel.h5'

model = tf.keras.models.load_model(
    model_path,
    custom_objects=None,
    compile=False
)

gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


gesture_to_label = {}
i = 0
for gesture in gestures:
  gesture_to_label[gesture] = i
  i+=1
# print(gesture_to_label)


label_to_gesture = {}
i = 0
for gesture in gestures:
  label_to_gesture[i] = gesture
  i+=1
# print(label_to_gesture)

x,y,w,h = 20, 150, 200, 250

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	if ret == False:
		continue

	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
	cv2.imshow("Frame", frame)

	img_section = frame[y:y+h,x:x+w]
	cv2.imshow("ROI", img_section)

	img = cv2.resize(img_section, (64, 64)) 
	cv2.imshow("Original", img) 

	img = img/255.0
	pred=model.predict(img.reshape(1, *img.shape))
	y_pred=np.argmax(pred, axis=1)[0]
	Y_pred = label_to_gesture[y_pred]

	cv2.putText(frame,'Predicted Label- ' + Y_pred,(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

print("Done Successfully")