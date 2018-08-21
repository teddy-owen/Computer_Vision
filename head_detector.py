#! /usr/bin/python2.7

import numpy as np
import cv2

def main():	
	face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')
	side_face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_profileface.xml')

	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	trailing_faces_count = 20
	trailing_faces = [0]*trailing_faces_count 

	while(True):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors = 3)
		side_faces = side_face_cascade.detectMultiScale(gray, 1.3, minNeighbors = 3)
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		for (x,y,w,h) in side_faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()