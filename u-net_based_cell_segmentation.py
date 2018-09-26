import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
import cv2

# Set some parameters
IMG_WIDTH = 444
IMG_HEIGHT = 444
IMG_CHANNELS = 3
TRAIN_PATH = '../dataprep/stage1_train/'
TEST_PATH = '../dataprep/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

def dopadding(img,pt,pb,pl,pr):
     height,width = img.shape[0:2]
     pt = int(pt)
     pb = int(pb)
     pl = int(pl)
     pr = int(pr)
     #print("pt"+pt)
     #print()
     padded_height = int(height + pt + pb)
     padded_width = int(width + pl + pr)
     paddedimg = np.zeros((padded_height,padded_width,3),dtype="uint8")

     #MAin image
     paddedimg[pt:pt+height,pl:pl+width] = img
     #Do padding_top
     top = img[0:pt,0:width]
     top = cv2.flip(top,0)#vertical f
     paddedimg[0:pt,pl:pl+width] = top
     #Do padding_bottom
     bottom = img[height-pb:height,0:width,:]
     bottom = cv2.flip(bottom,0)
     paddedimg[(padded_height - pb):padded_height,pl:pl+width,:] = bottom
     #Do leftpadding
     left = paddedimg[0:padded_height,pl:pl+pl,:]
     left = cv2.flip(left,1)
     paddedimg[0:padded_height,0:pl,:] = left
     #Do right padding
     right = paddedimg[0:padded_height,pl+width-pr:pl+width,:]
     right = cv2.flip(right,1)

     paddedimg[0:padded_height,(padded_width - pr):padded_width] = right
     #flippedpadded = cv2.flip(paddedimg,0)
     return paddedimg

def dopaddingtraining(img,pt,pb,pl,pr):
     height,width = img.shape[0:2]
     pt = int(pt)
     pb = int(pb)
     pl = int(pl)
     pr = int(pr)
     #print("pt"+pt)
     #print()
     padded_height = int(height + pt + pb)
     padded_width = int(width + pl + pr)
     paddedimg = np.zeros((padded_height,padded_width,3),dtype="uint8")

     paddedimg[pt:pt+height,pl:pl+width] = img
     #Do padding_top
     top = img[0:pt,0:width]
     top = cv2.flip(top,0)#vertical f
     paddedimg[0:pt,pl:pl+width] = top
     #Do padding_bottom
     bottom = img[height-pb:height,0:width,:]
     bottom = cv2.flip(bottom,0)
     paddedimg[(padded_height - pb):padded_height,pl:pl+width,:] = bottom
     #Do leftpadding
     left = paddedimg[0:padded_height,pl:pl+pl,:]
     left = cv2.flip(left,1)
     paddedimg[0:padded_height,0:pl,:] = left
     #Do right padding
     right = paddedimg[0:padded_height,pl+width-pr:pl+width,:]
     right = cv2.flip(right,1)

     paddedimg[0:padded_height,(padded_width - pr):padded_width] = right
     return paddedimg

def dopaddingmask(img,pt,pb,pl,pr):
     height,width = img.shape[0:2]
     pt = int(pt)
     pb = int(pb)
     pl = int(pl)
     pr = int(pr)
     #print("pt"+pt)
     #print()
     padded_height = int(height + pt + pb)
     padded_width = int(width + pl + pr)
     paddedimg = np.zeros((padded_height,padded_width),dtype="uint8")

     #MAin image
     paddedimg[pt:pt+height,pl:pl+width] = img
     #Do padding_top
     top = img[0:pt,0:width]
     top = cv2.flip(top,0)
     paddedimg[0:pt,pl:pl+width] = top
     #Do padding_bottom
     bottom = img[height-pb:height,0:width]
     bottom = cv2.flip(bottom,0)
     paddedimg[(padded_height - pb):padded_height,pl:pl+width] = bottom
     #Do leftpadding
     left = paddedimg[0:padded_height,pl:pl+pl]
     left = cv2.flip(left,1)
     paddedimg[0:padded_height,0:pl] = left
     #Do right padding
     right = paddedimg[0:padded_height,pl+width-pr:pl+width]
     right = cv2.flip(right,1)
     paddedimg[0:padded_height,(padded_width - pr):padded_width] = right
     return paddedimg

# Calculate buffer size needed for training images
totalcount = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    height,width = img.shape[0:2]
    vslices = int((height + 255)/256)
    hslices = int((width + 255)/256)
    totalcount += vslices*hslices

totalcount = 4*totalcount
# Get and resize train images and masks
X_train = np.zeros((totalcount, 444, 444, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((totalcount, 256, 256, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

count = 0
kernel = np.ones((3,3),np.uint8)
paddingsize = 94
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    trainingimg = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    act_height,act_width = trainingimg.shape[0:2]
    paddedtrainingimage1 = dopaddingtraining(trainingimg,paddingsize,paddingsize,paddingsize,paddingsize)
    height,width = paddedtrainingimage1.shape[0:2]
    mask1 = np.zeros((height, width, 1), dtype=np.bool)
    imgoriginal = np.zeros((act_height,act_width), np.uint8)
    imgdilated = np.zeros((act_height,act_width), np.uint8)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = cv2.imread(path + '/masks/' + mask_file,cv2.IMREAD_GRAYSCALE)
        mask_ = cv2.dilate(mask_,kernel)
        mask_ = cv2.erode(mask_,kernel)
        imgoriginal = cv2.addWeighted(imgoriginal,1.0,mask_,1,0)
        imgdilated = cv2.addWeighted(imgdilated,1.0,cv2.dilate(mask_,kernel),0.5,0)
    ret,imgthresholded = cv2.threshold(imgdilated,200, 255, cv2.THRESH_BINARY_INV)
    finalimage = cv2.bitwise_and(imgoriginal,imgthresholded)
    finalimage1 = dopaddingmask(finalimage,paddingsize,paddingsize,paddingsize,paddingsize)
    mask_ = np.expand_dims(finalimage1, axis=-1)
    mask1 = np.maximum(mask1,mask_)
    paddedmaskimage1 = (mask1 != 0)

    #X_train[n] = trainingimg
    #Y_train[n] = maskimage
    #paddedmaskimage = dopaddingmask(maskimage,paddingsize,paddingsize,paddingsize,paddingsize)
    #print (paddedimage.shape)
    pi_height,pi_width = paddedtrainingimage1.shape[0:2]
    vslices = int((trainingimg.shape[0] + 255)/256)
    hslices = int((trainingimg.shape[1] + 255)/256)
    startheight = 0
    for i in range(0,vslices):
        if (startheight + 444) > pi_height:
            startheight = pi_height - 444
        startwidth = 0
        for j in range(0,hslices):
            if (startwidth + 444) > pi_width:
                startwidth = pi_width - 444
            X_train[count] = paddedtrainingimage1[startheight:startheight+444,startwidth:startwidth+444]
            Y_train[count] = paddedmaskimage1[startheight+94:startheight+350,startwidth+94:startwidth+350]
            X_train[count + 1] = cv2.flip(X_train[count],0)
            Y_train[count + 1] = cv2.flip(X_train[count],0)
            X_train[count + 2] = cv2.flip(X_train[count],1)
            Y_train[count + 2] = cv2.flip(X_train[count],1)
            X_train[count + 3] = cv2.flip(X_train[count],-1)
            Y_train[count + 3] = cv2.flip(X_train[count],-1)
            count += 4
            startwidth += 256
        startheight += 256

# Get and resize test images
sizes_test = []
cropping = []
totalcount = 0
count = 0
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    height,width = img.shape[0:2]
    vslices = int((height + 255)/256)
    hslices = int((width + 255)/256)
    totalcount += vslices*hslices
    count += 1

print('Getting and cropping test images ... ')
sys.stdout.flush()
X_test = np.zeros((totalcount, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
count = 0
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    height, width = img.shape[0:2]
    sizes_test.append([img.shape[0], img.shape[1]])
    if img.shape[0] < 256:
        padding_top = int((444 - img.shape[0])/2)
        padding_bottom = int(444 - img.shape[0] - padding_top)
    else:
        padding_top = 94
        padding_bottom = 94

    if img.shape[1] < 256:
        padding_left = int((444 - img.shape[1])/2)
        padding_right = int(444 -  img.shape[1] - padding_left)
    else:
        padding_left = 94
        padding_right = 94

    paddedimage = dopadding(img,padding_top,padding_bottom,padding_left,padding_right)
    #print (paddedimage.shape)
    pi_height,pi_width = paddedimage.shape[0:2]
    cropping.append([padding_top-94,padding_left-94])
    vslices = int((height + 255)/256)
    hslices = int((width + 255)/256)
    startheight = 0
    for i in range(0,vslices):
        if (startheight + 444) > pi_height:
            startheight = pi_height - 444
        startwidth = 0
        for j in range(0,hslices):
            if (startwidth + 444) > pi_width:
                startwidth = pi_width - 444
            X_test[count] = paddedimage[startheight:startheight+444,startwidth:startwidth+444]
            count += 1
            startwidth += 256
        startheight += 256

print("Verify count"+str(count))
print('Done!')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (s)
#c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (p1)
#c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (p2)
#c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (p3)
#c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c4)
c4 = Dropout(0.5) (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (p4)
#c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c5)
c5 = Dropout(0.5) (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid') (c5)
c4c = Cropping2D(cropping = 4)(c4)
u6 = concatenate([u6, c4c])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid') (c6)
c3c = Cropping2D(cropping = 16)(c3)
u7 = concatenate([u7, c3c])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid') (c7)
c2c = Cropping2D(cropping = 40)(c2)
u8 = concatenate([u8, c2c])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='valid') (c8)
c1c = Cropping2D(cropping = 88)(c1)
u9 = concatenate([u9, c1c], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='valid') (c9)
c9c = Cropping2D(cropping = 2)(c9)
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9c)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.25, batch_size=4, epochs=100,
                    callbacks=[earlystopper, checkpointer],shuffle=True)
print("Done training")

# Predict on test data
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.75)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.75):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_merged = []
mapcount = 0
for i in range(0, len(test_ids)):
    startheight = 0
    vslices = int((sizes_test[i][0] + 255)/256)
    hslices = int((sizes_test[i][1] + 255)/256)
    imgremapheight = sizes_test[i][0]
    if imgremapheight < 256:
        imgremapheight = 256
    imgremapwidth = sizes_test[i][1]
    if imgremapwidth < 256:
        imgremapwidth = 256
    imageremapped = np.zeros((imgremapheight, imgremapwidth,1))
    for vcount in range(0,vslices):
        if (startheight + 256) > imgremapheight:
            startheight = imgremapheight - 256
        startwidth = 0
        for hcount in range(0,hslices):
            if (startwidth + 256) > imgremapwidth:
                startwidth = imgremapwidth - 256
            imageremapped[startheight:startheight+256,startwidth:startwidth+256] = preds_test[mapcount]
            mapcount+=1;
            startwidth += 256
        startheight += 256
    imgcropped = imageremapped[cropping[i][0]:cropping[i][0] + sizes_test[i][0],cropping[i][1]:cropping[i][1] + sizes_test[i][1]]
    preds_test_merged.append(imgcropped)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_merged[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)
print ("File done")
