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
data_path = PATH + '/Documents/FEI_Dataset'
data_dir_list = os.listdir(data_path)


# In[5]:


img_rows=28
img_cols=28
num_channel=3
num_epoch=10


# In[6]:


num_classes = 201


# In[7]:


img_data_list=[]


# In[8]:


for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img_resize=cv2.resize(input_img,(28,28))
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
labels[741:754]=50
labels[755:768]=51
labels[769:782]=52
labels[783:796]=53
labels[797:810]=54
labels[811:824]=55
labels[825:838]=56
labels[839:852]=57
labels[853:866]=58
labels[867:880]=59
labels[881:894]=60
labels[895:908]=61
labels[909:922]=62
labels[923:936]=63
labels[937:950]=64
labels[951:964]=65
labels[965:978]=66
labels[979:992]=67
labels[993:1006]=68
labels[1007:1020]=69
labels[1021:1034]=70
labels[1035:1048]=71
labels[1049:1062]=72
labels[1063:1076]=73
labels[1077:1090]=74
labels[1091:1104]=75
labels[1105:1118]=76
labels[1119:1132]=77
labels[1133:1146]=78
labels[1147:1160]=79
labels[1161:1174]=80
labels[1175:1188]=81
labels[1189:1202]=82
labels[1203:1216]=83
labels[1217:1230]=84
labels[1231:1244]=85
labels[1245:1258]=86
labels[1259:1272]=87
labels[1273:1286]=88
labels[1287:1300]=89
labels[1301:1314]=90
labels[1315:1328]=91
labels[1329:1342]=92
labels[1343:1356]=93
labels[1357:1370]=94
labels[1371:1384]=95
labels[1385:1398]=96
labels[1399:1412]=97
labels[1413:1426]=98
labels[1427:1440]=99
labels[1441:1455]=100
labels[1456:1470]=101
labels[1471:1485]=102
labels[1486:1500]=103
labels[1501:1515]=104
labels[1516:1530]=105
labels[1531:1545]=106
labels[1546:1560]=107
labels[1561:1575]=108
labels[1576:1590]=109
labels[1591:1605]=110
labels[1606:1620]=111
labels[1621:1635]=112
labels[1636:1650]=113
labels[1651:1665]=114
labels[1666:1680]=115
labels[1681:1695]=116
labels[1696:1710]=117
labels[1711:1725]=118
labels[1726:1740]=119
labels[1741:1754]=120
labels[1755:1768]=121
labels[1769:1782]=122
labels[1783:1796]=123
labels[1797:1810]=124
labels[1811:1824]=125
labels[1825:1838]=126
labels[1839:1852]=127
labels[1853:1866]=128
labels[1867:1880]=129
labels[1881:1894]=130
labels[1895:1908]=131
labels[1909:1922]=132
labels[1923:1936]=133
labels[1937:1950]=134
labels[1951:1964]=135
labels[1965:1978]=136
labels[1979:1992]=137
labels[1993:2006]=138
labels[2007:2020]=139
labels[2021:2034]=140
labels[2035:2048]=141
labels[2049:2062]=142
labels[2063:2076]=143
labels[2077:2090]=144
labels[2091:2104]=145
labels[2105:2118]=146
labels[2119:2132]=147
labels[2133:2146]=148
labels[2147:2160]=149
labels[2161:2174]=150
labels[2175:2188]=151
labels[2189:2202]=152
labels[2203:2216]=153
labels[2217:2230]=154
labels[2231:2244]=155
labels[2245:2258]=156
labels[2259:2272]=157
labels[2273:2286]=158
labels[2287:2300]=159
labels[2301:2314]=160
labels[2315:2328]=161
labels[2329:2342]=162
labels[2343:2356]=163
labels[2357:2370]=164
labels[2371:2384]=165
labels[2385:2398]=166
labels[2399:2412]=167
labels[2413:2426]=168
labels[2427:2440]=169
labels[2441:2454]=170
labels[2455:2468]=171
labels[2469:2482]=172
labels[2483:2496]=173
labels[2497:2510]=174
labels[2511:2524]=175
labels[2525:2538]=176
labels[2539:2552]=177
labels[2553:2566]=178
labels[2567:2580]=179
labels[2581:2594]=180
labels[2595:2608]=181
labels[2609:2622]=182
labels[2623:2636]=183
labels[2637:2650]=184
labels[2651:2664]=185
labels[2665:2678]=186
labels[2679:2692]=187
labels[2693:2706]=188
labels[2707:2720]=189
labels[2721:2734]=190
labels[2735:2748]=191
labels[2749:2762]=192
labels[2763:2776]=193
labels[2777:2789]=194
labels[2790:2802]=195
labels[2803:2815]=196
labels[2816:2828]=197
labels[2829:2841]=198
labels[2842:2854]=199
labels[2855:3027]=200
       
    
names = ['s001','s002','s003','s004','s005','s006','s007','s008','s009','s010','s011','s012','s013','s014','s015','s016','s017','s018','s019','s020','s021','s022','s023','s024','s025','s026','s027','s028','s029','s030','s031','s032','s033','s034','s035','s036','s037','s038','s039','s040','s041','s042','s043','s044','s045','s046','s047','s048','s049','s050','s051','s052','s053','s054','s055','s056','s057','s058','s059','s060','s061','s062','s063','s064','s065','s066','s067','s068','s069','s070','s071','s072','s073','s074','s075','s076','s077','s078','s079','s080','s081','s082','s083','s084','s085','s086','s087','s088','s089','s090','s091','s092','s093','s094','s095','s096','s097','s098','s099','s100','s101','s102','s103','s104','s105','s106','s107','s108','s109','s110','s111','s112','s113','s114','s115','s116','s117','s118','s119','s120','s121','s122','s123','s124','s125','s126','s127','s128','s129','s130','s131','s132','s133','s134','s135','s136','s137','s138','s139','s140','s141','s142','s143','s144','s145','s146','s147','s148','s149','s150','s151','s152','s153','s154','s155','s156','s157','s158','s159','s160','s161','s162','s163','s164','s165','s166','s167','s168','s169','s170','s171','s172','s173','s174','s175','s176','s177','s178','s179','s180','s181','s182','s183','s184','s185','s186','s187','s188','s189','s190','s191','s192','s193','s194','s195','s196','s197','s198','s199','s200','s201_nomatch']


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
model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(3,28,28), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(3,28,28), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(2000))

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


hist = model.fit(X_train, y_train, batch_size=128, epochs=49, verbose=1, validation_data=(X_test, y_test))


# In[38]:


from keras import callbacks

filename='model_train_FEI28.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)


# In[ ]:





# In[39]:


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





# In[31]:


#testing image 1


# In[25]:


test_image = cv2.imread('Documents/gt_test/2.jpg')
test_image=cv2.resize(test_image, (28,28))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.#shape)
   


# In[33]:


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


# In[34]:


print((model.predict(test_image)))
print(model.predict_classes(test_image))


# In[34]:


#testing image 2


# In[30]:


test_image = cv2.imread('Documents/gt_test/2.jpg')
test_image=cv2.resize(test_image, (28,28))
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




