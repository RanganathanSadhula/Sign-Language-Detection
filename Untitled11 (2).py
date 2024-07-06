#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train=pd.read_csv('C:\\Users\\91701\\Downloads\\sign_mnist_train.csv')
test=pd.read_csv('C:\\Users\\91701\\Downloads\\sign_mnist_train.csv')


# In[3]:


train.head()


# In[4]:


labels=train['label'].values


# In[5]:


unique_val=np.array(labels)
np.unique(unique_val)


# In[6]:


plt.figure(figsize=(18,2))
sns.countplot(x=labels)


# In[7]:


train.drop('label',axis=1,inplace=True)


# In[8]:


images=train.values
images=np.array([np.reshape(i,(28,28))for i in images])
images=np.array([i.flatten() for i in images])


# In[9]:


from sklearn.preprocessing import LabelBinarizer
label_binrizer=LabelBinarizer()
labels=label_binrizer.fit_transform(labels)


# In[10]:


labels


# In[11]:


index=2
print(labels[index])
plt.imshow(images[index].reshape(28,28))


# In[12]:


import cv2
import numpy as np


# In[13]:


for i in range(0,10):
    rand=np.random.randint(0, len(images))
    input_im=images[rand]
    
    sample=input_im.reshape(28,28).astype(np.uint8)
    sample=cv2.resize(sample,None,fx=10,fy=10,interpolation=cv2.INTER_CUBIC)
    cv2.imshow("sample image",sample)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()   
    
     


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.3,random_state=101)


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
batch_size=128
num_classes=24
epochs=10


# In[16]:


x_train=x_train/255
x_test=x_test/255


# In[17]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
plt.imshow(x_train[0].reshape(28,28))


# In[18]:


plt.imshow(x_train[0].reshape(28,28))


# In[19]:


from tensorflow.keras.layers import Conv2D ,MaxPooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam

model =Sequential()
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes,activation='softmax'))


# In[20]:


model.compile(loss='categorical_crossentropy',
             optimizer=Adam(),
             metrics=['accuracy'])
print(model.summary())


# In[21]:


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size)


# In[22]:


model.save("sign_mnist_cnn_50_Epochs.keras")
print("Model saved")


# In[23]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accurachy')
plt.legend(['train','test'])

plt.show()


# In[24]:


test_labels=test['label']
test.drop('label',axis=1,inplace=True)

test_images=test.values
test_images=np.array([np.reshape(i,(28,28)) for  i in test_images])
test_images=np.array([i.flatten() for i in test_images])

test_labels=label_binrizer.fit_transform(test_labels)

test_images=test_images.reshape(test_images.shape[0],28,28,1)

test_images.shape
y_pred=model.predict(test_images)


# In[25]:


from sklearn.metrics import accuracy_score

accuracy_score(test_labels,y_pred.round())


# In[26]:


def getLetter(result):
    classLabels={0:'A',
                 1:'B',
                 2:'C',
                 3:'D',
                 4:'E',
                 5:'F',
                 6:'G',
                 7:'H',
                 8:'I',
                 9:'J',
                 10:'K',
                 11:'L',
                 12:'M',
                 13:'N',
                 14:'O',
                 15:'P',
                 16:'Q',
                 17:'R',
                 18:'S',
                 19:'T',
                 20:'U',
                 21:'V',
                 22:'W',
                 23:'X'}
    try:
        res=int(result)
        return classLabels[res]
    except:
        return "Error"


# In[ ]:





# In[27]:


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    else:
        pass


# In[ ]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    cv2.imshow('roi scaled and gray', roi)
    
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)
    
    roi = roi.reshape(1, 28, 28, 1)
    
    # Using the 'predict' method
    prediction = model.predict(roi)
    predicted_class = np.argmax(prediction)
    result = str(predicted_class)
    
    cv2.putText(copy, getLetter(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


import cv2
import numpy as np
import math


vid = cv2.VideoCapture(0)

while True:
    flag, imgFlip = vid.read()
    img = cv2.flip(imgFlip,cv2.COLOR_BGR2GRAY)

    
    cv2.rectangle(img, (100,100), (300,300), (0,255,0), 0)
    imgCrop = img[100:300, 100:300]


    imgBlur = cv2.GaussianBlur(imgCrop, (3,3), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    lower = np.array([2,0,0])
    upper = np.array([20,255,255])
    mask = cv2.inRange(imgHSV, lower, upper)

   
    kernel = np.ones((5,5))

   
    dilation = cv2.dilate(mask,kernel, iterations=1)
    erosion = cv2.erode(dilation,kernel, iterations=1)

    filtered_img = cv2.GaussianBlur(erosion, (3,3), 0)
    ret, imgBin = cv2.threshold(filtered_img, 127, 255, 0)


    contours, hierarchy = cv2.findContours(imgBin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key = lambda x: cv2.contourArea(x))

       
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imgCrop, (x,y), (x+w,y+h), (0,0,255), 0)

     
        con_hull = cv2.convexHull(contour)


       
        con_hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, con_hull)
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

           
            if angle<=90:
                count_defects+=1
                cv2.circle(imgCrop, far, 2, [0,0,255], -1)

            cv2.line(imgCrop, start, end, [0,255,0], 2)

        if count_defects == 0:
            cv2.putText(img, "ONE", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
        elif count_defects == 1:
            cv2.putText(img, "TWO", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 2:
            cv2.putText(img, "THREE", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 3:
            cv2.putText(img, "FOUR", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 4:
            cv2.putText(img, "FIVE", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        else:
            pass

    except:
        pass

   

    cv2.imshow("Gesture", img)
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




