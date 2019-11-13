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
data_path = PATH + '/Documents/Face_Recog'
data_dir_list = os.listdir(data_path)


# In[5]:


img_rows=14
img_cols=14
num_channel=3
num_epoch=50


# In[6]:


num_classes = 376


# In[7]:


img_data_list=[]


# In[8]:


for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img_resize=cv2.resize(input_img,(14,14))
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
 
    
labels[0:20]=0
labels[21:40]=1
labels[41:60]=2
labels[61:80]=3
labels[81:100]=4
labels[101:120]=5
labels[121:140]=6
labels[141:160]=7
labels[161:180]=8
labels[181:200]=9
labels[201:220]=10
labels[221:240]=11
labels[241:260]=12
labels[261:280]=13
labels[281:300]=14
labels[301:320]=15
labels[321:340]=16
labels[341:360]=17
labels[361:380]=18
labels[381:400]=19
labels[401:420]=20
labels[421:440]=21
labels[441:460]=22
labels[461:480]=23
labels[481:500]=24
labels[501:520]=25
labels[521:540]=26
labels[541:560]=27
labels[561:580]=28
labels[581:600]=29
labels[601:620]=30
labels[621:640]=31
labels[641:660]=32
labels[661:680]=33
labels[681:700]=34
labels[701:720]=35
labels[721:740]=36
labels[741:760]=37
labels[761:780]=38
labels[781:800]=39
labels[801:820]=40
labels[821:840]=41
labels[841:860]=42
labels[861:880]=43
labels[881:900]=44
labels[901:920]=45
labels[921:940]=46
labels[941:960]=47
labels[961:980]=48
labels[981:1000]=49
labels[1001:1020]=50
labels[1021:1040]=51
labels[1041:1060]=52
labels[1061:1080]=53
labels[1081:1100]=54
labels[1101:1120]=55
labels[1121:1140]=56
labels[1141:1160]=57
labels[1161:1180]=58
labels[1181:1200]=59
labels[1201:1220]=60
labels[1221:1240]=61
labels[1241:1260]=62
labels[1261:1280]=63
labels[1281:1300]=64
labels[1301:1320]=65
labels[1321:1340]=66
labels[1341:1360]=67
labels[1361:1380]=68
labels[1381:1400]=69
labels[1401:1420]=70
labels[1421:1440]=71
labels[1441:1460]=72
labels[1461:1480]=73
labels[1481:1500]=74
labels[1501:1520]=75
labels[1521:1540]=76
labels[1541:1560]=77
labels[1561:1580]=78
labels[1581:1600]=79
labels[1601:1620]=80
labels[1621:1640]=81
labels[1641:1660]=82
labels[1661:1680]=83
labels[1681:1700]=84
labels[1701:1720]=85
labels[1721:1740]=86
labels[1741:1760]=87
labels[1761:1780]=88
labels[1781:1800]=89
labels[1801:1820]=90
labels[1821:1840]=91
labels[1841:1860]=92
labels[1861:1880]=93
labels[1881:1900]=94
labels[1901:1920]=95
labels[1921:1940]=96
labels[1941:1960]=97
labels[1961:1980]=98
labels[1981:2000]=99
labels[2001:2020]=100
labels[2021:2040]=101
labels[2041:2060]=102
labels[2061:2080]=103
labels[2081:2100]=104
labels[2101:2120]=105
labels[2121:2140]=106
labels[2141:2160]=107
labels[2161:2180]=108
labels[2181:2200]=109
labels[2201:2220]=110
labels[2221:2240]=111
labels[2241:2260]=112
labels[2261:2280]=113
labels[2281:2300]=114
labels[2301:2320]=115
labels[2321:2340]=116
labels[2341:2360]=117
labels[2361:2380]=118
labels[2381:2400]=119
labels[2401:2420]=120
labels[2421:2440]=121
labels[2441:2460]=122
labels[2461:2480]=123
labels[2481:2500]=124
labels[2501:2520]=125
labels[2521:2540]=126
labels[2541:2560]=127
labels[2561:2580]=128
labels[2581:2600]=129
labels[2601:2620]=130
labels[2621:2640]=131
labels[2641:2660]=132
labels[2661:2680]=133
labels[2681:2700]=134
labels[2701:2720]=135
labels[2721:2740]=136
labels[2741:2760]=137
labels[2761:2780]=138
labels[2781:2800]=139
labels[2801:2820]=140
labels[2821:2840]=141
labels[2841:2860]=142
labels[2861:2880]=143
labels[2881:2900]=144
labels[2901:2920]=145
labels[2921:2940]=146
labels[2941:2960]=147
labels[2961:2980]=148
labels[2981:3000]=149
labels[3001:3020]=150
labels[3021:3040]=151
labels[3041:3060]=152
labels[3061:3080]=153
labels[3081:3100]=154
labels[3101:3120]=155
labels[3121:3140]=156
labels[3141:3160]=157
labels[3161:3180]=158
labels[3181:3200]=159
labels[3201:3220]=160
labels[3221:3240]=161
labels[3241:3260]=162
labels[3261:3280]=163
labels[3281:3300]=164
labels[3301:3320]=165
labels[3321:3340]=166
labels[3341:3360]=167
labels[3361:3380]=168
labels[3381:3400]=169
labels[3401:3420]=170
labels[3421:3440]=171
labels[3441:3460]=172
labels[3461:3480]=173
labels[3481:3500]=174
labels[3501:3520]=175
labels[3521:3540]=176
labels[3541:3560]=177
labels[3561:3580]=178
labels[3581:3600]=179
labels[3601:3620]=180
labels[3621:3640]=181
labels[3641:3660]=182
labels[3661:3680]=183
labels[3681:3700]=184
labels[3701:3720]=185
labels[3721:3740]=186
labels[3741:3760]=187
labels[3761:3780]=188
labels[3781:3800]=189
labels[3801:3820]=190
labels[3821:3840]=191
labels[3841:3860]=192
labels[3861:3880]=193
labels[3881:3900]=194
labels[3901:3920]=195
labels[3921:3940]=196
labels[3941:3960]=197
labels[3961:3980]=198
labels[3981:4000]=199
labels[4001:4020]=200
labels[4021:4040]=201
labels[4041:4060]=202
labels[4061:4080]=203
labels[4081:4100]=204
labels[4101:4120]=205
labels[4121:4140]=206
labels[4141:4160]=207
labels[4161:4180]=208
labels[4181:4200]=209
labels[4201:4220]=210
labels[4221:4240]=211
labels[4241:4260]=212
labels[4261:4280]=213
labels[4281:4300]=214
labels[4301:4320]=215
labels[4321:4340]=216
labels[4341:4360]=217
labels[4361:4380]=218
labels[4381:4400]=219
labels[4401:4420]=220
labels[4421:4440]=221
labels[4441:4460]=222
labels[4461:4480]=223
labels[4481:4500]=224
labels[4501:4520]=225
labels[4521:4540]=226
labels[4541:4560]=227
labels[4561:4580]=228
labels[4581:4600]=229
labels[4601:4620]=230
labels[4621:4640]=231
labels[4641:4660]=232
labels[4661:4680]=233
labels[4681:4700]=234
labels[4701:4720]=235
labels[4721:4740]=236
labels[4741:4760]=237
labels[4761:4780]=238
labels[4781:4800]=239
labels[4801:4820]=240
labels[4821:4840]=241
labels[4841:4860]=242
labels[4861:4880]=243
labels[4881:4900]=244
labels[4901:4920]=245
labels[4921:4940]=246
labels[4941:4960]=247
labels[4961:4980]=248
labels[4981:5000]=249
labels[5001:5020]=250
labels[5021:5040]=251
labels[5041:5060]=252
labels[5061:5080]=253
labels[5081:5100]=254
labels[5101:5120]=255
labels[5121:5140]=256
labels[5141:5160]=257
labels[5161:5180]=258
labels[5181:5200]=259
labels[5201:5220]=260
labels[5221:5240]=261
labels[5241:5260]=262
labels[5261:5280]=263
labels[5281:5300]=264
labels[5301:5320]=265
labels[5321:5340]=266
labels[5341:5360]=267
labels[5361:5380]=268
labels[5381:5400]=269
labels[5401:5420]=270
labels[5421:5440]=271
labels[5441:5460]=272
labels[5461:5480]=273
labels[5481:5500]=274
labels[5501:5520]=275
labels[5521:5540]=276
labels[5541:5560]=277
labels[5561:5580]=278
labels[5581:5600]=279
labels[5601:5620]=280
labels[5621:5640]=281
labels[5641:5660]=282
labels[5661:5680]=283
labels[5681:5700]=284
labels[5701:5720]=285
labels[5721:5740]=286
labels[5741:5760]=287
labels[5761:5780]=288
labels[5781:5800]=289
labels[5801:5820]=290
labels[5821:5840]=291
labels[5841:5860]=292
labels[5861:5880]=293
labels[5881:5900]=294
labels[5901:5920]=295
labels[5921:5940]=296
labels[5941:5960]=297
labels[5961:5980]=298
labels[5981:6000]=299
labels[6001:6020]=300
labels[6021:6040]=301
labels[6041:6060]=302
labels[6061:6080]=303
labels[6081:6100]=304
labels[6101:6120]=305
labels[6121:6140]=306
labels[6141:6160]=307
labels[6161:6180]=308
labels[6181:6200]=309
labels[6201:6220]=310
labels[6221:6240]=311
labels[6241:6260]=312
labels[6261:6280]=313
labels[6281:6300]=314
labels[6301:6320]=315
labels[6321:6340]=316
labels[6341:6360]=317
labels[6361:6380]=318
labels[6381:6400]=319
labels[6401:6420]=320
labels[6421:6440]=321
labels[6441:6460]=322
labels[6461:6480]=323
labels[6481:6500]=324
labels[6501:6520]=325
labels[6521:6540]=326
labels[6541:6560]=327
labels[6561:6580]=328
labels[6581:6600]=329
labels[6601:6620]=330
labels[6621:6640]=331
labels[6641:6660]=332
labels[6661:6680]=333
labels[6681:6700]=334
labels[6701:6720]=335
labels[6721:6740]=336
labels[6741:6760]=337
labels[6761:6780]=338
labels[6781:6800]=339
labels[6801:6820]=340
labels[6821:6840]=341
labels[6841:6860]=342
labels[6861:6880]=343
labels[6881:6900]=344
labels[6901:6920]=345
labels[6921:6940]=346
labels[6941:6960]=347
labels[6961:6980]=348
labels[6981:7000]=349
labels[7001:7020]=350
labels[7021:7040]=351
labels[7041:7060]=352
labels[7061:7080]=353
labels[7081:7100]=354
labels[7101:7120]=355
labels[7121:7140]=356
labels[7141:7160]=357
labels[7161:7180]=358
labels[7181:7200]=359
labels[7201:7220]=360
labels[7221:7240]=361
labels[7241:7260]=362
labels[7261:7280]=363
labels[7281:7300]=364
labels[7301:7320]=365
labels[7321:7340]=366
labels[7341:7360]=367
labels[7361:7380]=368
labels[7381:7400]=369
labels[7401:7420]=370
labels[7421:7440]=371
labels[7441:7460]=372
labels[7461:7480]=373
labels[7481:7500]=374
labels[7501:7673]=375


       
    
names = ['s001','s002','s003','s004','s005','s006','s007','s008','s009','s010','s011','s012','s013','s014','s015','s016','s017','s018','s019','s020','s021','s022','s023','s024','s025','s026','s027','s028','s029','s030','s031','s032','s033','s034','s035','s036','s037','s038','s039','s040','s041','s042','s043','s044','s045','s046','s047','s048','s049','s050','s051','s052','s053','s054','s055','s056','s057','s058','s059','s060','s061','s062','s063','s064','s065','s066','s067','s068','s069','s070','s071','s072','s073','s074','s075','s076','s077','s078','s079','s080','s081','s082','s083','s084','s085','s086','s087','s088','s089','s090','s091','s092','s093','s094','s095','s096','s097','s098','s099','s100','s101','s102','s103','s104','s105','s106','s107','s108','s109','s110','s111','s112','s113','s114','s115','s116','s117','s118','s119','s120','s121','s122','s123','s124','s125','s126','s127','s128','s129','s130','s131','s132','s133','s134','s135','s136','s137','s138','s139','s140','s141','s142','s143','s144','s145','s146','s147','s148','s149','s150','s151','s152','s153','s154','s155','s156','s157','s158','s159','s160','s161','s162','s163','s164','s165','s166','s167','s168','s169','s170','s171','s172','s173','s174','s175','s176','s177','s178','s179','s180','s181','s182','s183','s184','s185','s186','s187','s188','s189','s190','s191','s192','s193','s194','s195','s196','s197','s198','s199','s200','s201','s202','s203','s204','s205','s206','s207','s208','s209','s210','s211','s212','s213','s214','s215','s216','s217','s218','s219','s220','s221','s222','s223','s224','s225','s226','s227','s228','s229','s230','s231','s232','s233','s234','s235','s236','s237','s238','s239','s240','s241','s242','s243','s244','s245','s246','s247','s248','s249','s250','s251','s252','s253','s254','s255','s256','s257','s258','s259','s260','s261','s262','s263','s264','s265','s266','s267','s268','s269','s270','s271','s272','s273','s274','s275','s276','s277','s278','s279','s280','s281','s282','s283','s284','s285','s286','s287','s288','s289','s290','s291','s292','s293','s294','s295','s296','s297','s298','s299','s300','s301','s302','s303','s304','s305','s306','s307','s308','s309','s310','s311','s312','s313','s314','s315','s316','s317','s318','s319','s320','s321','s322','s323','s324','s325','s326','s327','s328','s329','s330','s331','s332','s333','s334','s335','s336','s337','s338','s339','s340','s341','s342','s343','s344','s345','s346','s347','s348','s349','s350','s351','s352','s353','s354','s355','s356','s357','s358','s359','s360','s361','s362','s363','s364','s365','s366','s367','s368','s369','s370','s371','s372','s373','s374','s375','s376_nomatch']


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
model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(3,14,14), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(3,14,14), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(6000))

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


hist = model.fit(X_train, y_train, batch_size=256, epochs=50, verbose=1, validation_data=(X_test, y_test))


# In[21]:


from keras import callbacks

filename='model_train_Frec14.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)


# In[ ]:





# In[22]:


early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]


# In[23]:


hist = model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)


# In[24]:


train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(4)


# In[25]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# In[26]:


test_image = X_test[0:1]
print (test_image.shape)


# In[27]:


print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])


# In[ ]:





# In[28]:


#testing image 1


# In[29]:


test_image = cv2.imread('Documents/gt_test/2.jpg')
test_image=cv2.resize(test_image, (56,56))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   


# In[30]:


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


# In[31]:


print((model.predict(test_image)))
print(model.predict_classes(test_image))


# In[33]:


#testing image 2


# In[32]:


test_image = cv2.imread('Documents/gt_test/fei8.jpg')
test_image=cv2.resize(test_image, (56,56))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




