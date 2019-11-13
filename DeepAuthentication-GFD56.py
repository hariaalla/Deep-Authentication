#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[2]:


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[3]:


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


# In[4]:


PATH = os.getcwd()
# Define data path
data_path = PATH + '/Documents/gt_db'
data_dir_list = os.listdir(data_path)


# In[5]:


img_rows=56
img_cols=56
num_channel=3
num_epoch=10


# In[6]:


num_classes = 51


# In[7]:


img_data_list=[]


# In[8]:


for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img_resize=cv2.resize(input_img,(56,56))
		img_data_list.append(input_img_resize)


# In[9]:


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)


# In[10]:


from keras import backend as K
K.set_image_dim_ordering('th')


# In[11]:


if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)


# In[12]:


num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
 
    
labels[0:14]=0
labels[15:28]=1
labels[29:42]=2
labels[43:56]=3
labels[57:70]=4
labels[71:84]=5
labels[85:98]=6
labels[99:112]=7
labels[113:126]=8
labels[127:140]=9
labels[141:155]=10
labels[156:170]=11
labels[171:185]=12
labels[186:200]=13
labels[201:215]=14
labels[216:230]=15
labels[231:245]=16
labels[246:260]=17
labels[261:275]=18
labels[276:290]=19
labels[291:305]=20
labels[306:320]=21
labels[321:335]=22
labels[336:350]=23
labels[351:365]=24
labels[366:380]=25
labels[381:395]=26
labels[396:410]=27
labels[411:425]=28
labels[426:440]=29
labels[441:455]=30
labels[456:470]=31
labels[471:485]=32
labels[486:500]=33
labels[501:515]=34
labels[516:530]=35
labels[531:545]=36
labels[546:560]=37
labels[561:575]=38
labels[576:590]=39
labels[591:605]=40
labels[606:620]=41
labels[621:635]=42
labels[636:650]=43
labels[651:665]=44
labels[666:680]=45
labels[681:695]=46
labels[696:710]=47
labels[711:725]=48
labels[726:740]=49
labels[741:913]=50
    
    
names = ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s27','s28','s29','s30','s31','s32','s33','s34','s35','s36','s37','s38','s39','s40','s41','s42','s43','s44','s45','s46','s47','s48','s49','s50','s51_nomatch']


# In[13]:


Y = np_utils.to_categorical(labels, num_classes)


# In[14]:


x,y = shuffle(img_data,Y, random_state=2)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


# In[16]:


input_shape=img_data[0].shape


# In[17]:


model = Sequential()
model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(3,56,56), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(3,56,56), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512))

model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[18]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])


# In[19]:


model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable


# In[20]:


hist = model.fit(X_train, y_train, batch_size=32, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))


# In[23]:


from keras import callbacks

filename='model_train_new1.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)


# In[ ]:





# In[24]:


early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]


# In[25]:


hist = model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)


# In[21]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# In[22]:


test_image = X_test[0:1]
print (test_image.shape)


# In[23]:


print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])


# In[ ]:





# In[30]:


#testing image 1


# In[30]:


test_image = cv2.imread('Documents/gt_test/2.jpg')
test_image=cv2.resize(test_image, (56,56))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   


# In[31]:


if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)


# In[32]:


print((model.predict(test_image)))
print(model.predict_classes(test_image))


# In[34]:


#testing image 2


# In[38]:


test_image = cv2.imread('Documents/gt_test/1.jpg')
test_image=cv2.resize(test_image, (56,56))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)


# In[39]:


if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)


# In[40]:


print((model.predict(test_image)))
print(model.predict_classes(test_image))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




