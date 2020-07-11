#Recognise Faces using some classification algorithm- like logistic, KNN, SVM etc.


#1 load the training data(numpy arrays of all the persons)
        #x-values are stored in the numpy arrays
        #y-values we need to assign for each person
#2. Read a video stream using openCV
#3. extract faces out of it(Testing)
#4. Use knn to find the prediction of face(int)
#5. Map the predicted id to name of the user
#6. Display the predictions on the screen - bounding box and name
import numpy as np
import cv2
import os

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(train,test,k=5):
    vals=[]
    m=train.shape[0]
    for i in range(m):
    	#Get the vector and the label
    	ix=train[i,:-1]
    	iy=train[i,-1]
    	#Compute distance from test point
    	d=dist(test,ix)
    	vals.append([d,iy])
		
		
		
	    
	#Sort based on distance and get top k
    dk=sorted(vals,key=lambda x:x[0])[:k]
    #Retrieves on the labels
    labels=np.array(dk)[:,-1]

    #Get frequencies of each label
    output=np.unique(labels,return_counts=True)
    #Find maximum frequency and corresponding label
    index=np.argmax(output[1])
        
    return output[0][index]


cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]

dataset_path='./data/'
labels=[]

class_id=0#Labels for given file
names={}#Mapping between id and name

#Data preparation

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#Create a mapping between class_id and name

		names[class_id]=fx[:-4]#Taking all characters except .npy

		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)

		#Create labels for the class
		target=class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

#Testing

while True:
	ret,frame=cap.read()

	if ret==False:
		continue
	faces=face_cascade.detectMultiScale(frame,1.3,5)

	for (x,y,w,h) in faces:
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

        #Predicted label (out)
		out=knn(trainset,face_section.flatten())

		#Display on the screen the name and rectangle around it

		pred_name=names[int(out)]

		
 
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key=cv2.waitKey(1)&0xFF
	if(key==ord('q')):
		break;
cap.release()
cv2.destroyAllWindows()


