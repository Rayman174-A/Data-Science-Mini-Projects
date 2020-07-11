# Read a video stream from camera(frame by frame)
import cv2

cap=cv2.VideoCapture(0)# default webcam
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame=cap.read()#ret is a boolean value representing capture of image
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	
    
	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(125,0,130),2)


	cv2.imshow("Video Frame",frame)
	#cv2.imshow("Gray Frame",gray_frame)

	#Wait for user input. Like user presses q and then you will stop
	key_pressed=cv2.waitKey(1) & 0xFF ##0xFF has 8 ones(converts 32 bit into 8 bit)
	if key_pressed ==ord('q'):#Ord return ASCII Value
		break

cap.release()
cv2.destroyAllWindows()
