#Write a python script that captures images from your webcam video stream
#Extracts all faces from the image frame(using haarcascades)
#Stores the face information into numpy arrays

#1. Read and show video stream, capture images
#2. Detect faces and show bounding box(haarcascade)
#3. Flatten the largest face image(gray scale) and store in a numpy array
#4. Repeat the same for multiple people to generate training data
import cv2
import numpy
#Initialise camera
cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='./data/'

file_name=input("Enter the name of person: ")
while True:
	ret,frame=cap.read()
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	
	faces=face_cascade.detectMultiScale(frame,1.3,5)#Value of scaling parameter and number of scales
	faces=sorted(faces,key=lambda f:f[2]*f[3])
	#Picking the last face because it is the largest face according to area which is given by f[2]*f[3] i.e width*height(w*h)


	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract(Crop out the required path)
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
		cv2.imshow("faceSection",face_section)

		skip+=1

		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
			pass

	cv2.imshow("Frame",frame)
	
	

	

	key_pressed=cv2.waitKey(1)& 0xFF
	if key_pressed==ord('q'):
		break
# Convert our face list array into numpy array
face_data=numpy.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save data into file system
numpy.save(dataset_path+file_name+'.npy',face_data)

print("Data successfully saved at "+dataset_path+file_name+'.npy',face_data)
cap.release()
cv2.destroyAllWindows()
