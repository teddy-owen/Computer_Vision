#! /usr/bin/python2.7

import numpy as np
import cv2

def main():	
	body_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_upperbody.xml')

	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	trailing_body_count = 20
	trailing_bodies = [0]*trailing_body_count 

	while(True):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		bodies = body_cascade.detectMultiScale(gray, 1.3, minNeighbors = 2)
		body_count = len(bodies)
		trailing_bodies.append(body_count)
		trailing_bodies.pop(0)
		trailing_med = np.median(trailing_bodies)
		print(trailing_med)
		message = ""
		if trailing_med > 0:
			message = "Person Detected"	
		cv2.putText(frame,message,(50,50), font, 1,(0,0,255),2,cv2.LINE_AA)
		for (x,y,w,h) in bodies:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()