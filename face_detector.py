#! /usr/bin/python2.7

import numpy as np
import cv2
from time import sleep

def main():	
	face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')

	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	trailing_faces_count = 20
	trailing_faces = [0]*trailing_faces_count 

	while(True):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors = 3)
		face_count = len(faces)
		trailing_faces.append(face_count)
		trailing_faces.pop(0)
		trailing_med = np.median(trailing_faces)
		print(trailing_med)
		message = ""
		if trailing_med > 2:
			 message = "I see all of you"
		elif trailing_med > 1:
			message = "I see you both"	
		elif trailing_med > 0:
			message = "I see you"
		cv2.putText(frame,message,(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()