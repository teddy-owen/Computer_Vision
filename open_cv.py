import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('/home/teddy/Documents/software_dev/Computer_Vision/classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/teddy/Documents/software_dev/Computer_Vision/classifiers/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/home/teddy/Documents/software_dev/Computer_Vision/classifiers/haarcascade_smile.xml')
# img = cv2.imread('/home/teddy/Downloads/group.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
trailing_faces_count = 50
trailing_faces = [0]*trailing_faces_count 
# print(trailing_faces)
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	face_count = len(faces)
	trailing_faces.append(face_count)
	trailing_faces.pop(0)
	trailing_med = np.median(trailing_faces)
	# print(trailing_med)
	print(trailing_faces)
	message = ""
	if trailing_med > 2:
		 message = "I see all of you!"
	elif trailing_med > 1:
		message = "I see you both!"	
	elif trailing_med > 0:
		message = "I see you!"	
	# if  face_count > 0:
	cv2.putText(frame,message,(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		smile = smile_cascade.detectMultiScale(roi_gray,minNeighbors = 10)
		eyes = eye_cascade.detectMultiScale(roi_gray,minNeighbors = 40)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		for (ex,ey,ew,eh) in smile:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()