#! /usr/bin/python2.7

import numpy as np
import cv2

def main():	
	face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')
	smile_cascade = cv2.CascadeClassifier('/home/teddy/Documents/software_dev/Computer_Vision/classifiers/haarcascade_smile.xml')

	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	trailing_smiles_count = 20
	trailing_smiles= [0]*trailing_smiles_count 

	while(True):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors = 3)
		for (x,y,w,h) in faces:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			smiles = smile_cascade.detectMultiScale(roi_gray,minNeighbors = 10)
			smile_count = len(smiles)
			trailing_smiles.append(smile_count)
			trailing_smiles.pop(0)
			trailing_med = np.median(trailing_smiles)
			print(trailing_med)
			message = ""
			if trailing_med > 1:
				message = "I see smiles"	
			elif trailing_med > 0:
				message = "I see a smile"
			cv2.putText(frame,message,(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
			for (ex,ey,ew,eh) in smiles:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()