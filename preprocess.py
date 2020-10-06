from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from globalSetting import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans
import random as r
import SimpleITK as sitk
import matplotlib.animation as animation
import glob
import cv2
import time
import csv
import os
from keras.utils import np_utils
#from AUnet import *

img_size = get_img_size()
x_file_names = get_x_file_names()
y_file_names = get_y_file_names()
aug_methods = get_aug_methods()
csvHeaders = get_csvHeaders()


def n4itk(img):  # must input with sitk img object
    img = sitk.GetImageFromArray(img)
    img = sitk.Cast(img, sitk.sitkFloat32)
    img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))
    corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
    img_ = sitk.GetArrayFromImage(corrected_img)
    return img_


def resize(img):
    global index
    # plt.subplot(index)
    # index = index + 1
    # plt.imshow(img[47], cmap = 'gray')
    img_ = []
    for slice in range(len(img)):
        img_tmp = cv2.resize(img[slice], resize, interpolation=cv2.INTER_NEAREST)
        img_tmp = img_tmp.astype('uint8')
        img_.append(img_tmp)
    img = np.array(img_)
    return img
    # print(img.shape, np.unique(img))
    # img = trans.rescale(img, 0.5, mode='reflect')


def resize_slice(img, size):
    global index
    # plt.subplot(index)
    # index = index + 1
    # plt.imshow(img[47], cmap = 'gray')
    img_tmp = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    return img_tmp
    # print(img.shape, np.unique(img))
    # img = trans.rescale(img, 0.5, mode='reflect')


def data_generator():  # input img must be rank 4
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=25,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        # validation_split=0.2,
        zoom_range=False)
    return datagen


def augmentation_4_classification(x, y, k):  # input img must be rank 4
    if k == 0:
        return x, y
    start_time = time.time()
    datagen = data_generator()
    HGGsum = int(sum(y))
    x_ = x.copy()
    y_ = y.copy()
    x1 = x[:HGGsum]
    x2 = x[HGGsum:]
    n = k
    for x_batch in datagen.flow(x1, batch_size=1, seed=1000):
        x_ = np.vstack([x_, x_batch])
        y_ = np.append(y_, 1)
        n = n - 1
        if n == 0:
            break
    n = k
    elapsed_time = time.time() - start_time
    for x_batch in datagen.flow(x2, batch_size=1, seed=1000):
        x_ = np.vstack([x_, x_batch])
        y_ = np.append(y_, 0)
        n = n - 1
        if n == 0:
            break
    elapsed_time = time.time() - start_time
    print("Augmentation took time:", elapsed_time)
    return x_, y_


def augmentation_methods(x, y, n):
    aug = aug_methods[n]
    if n == 0:
        return x, y, aug
    elif n == 1:
        x_, y_ = augmentation(x, y, 335)
        return x_, y_, aug
    elif n == 2:
        x_, y_ = augmentation(x, y, 1000)  # 3350
        return x_, y_, aug


def augmentation(x, y, n):  # input img must be rank 4
    if n == 0:
        return x, y
    datagen = data_generator()
    x2 = x.copy()
    y2 = y.copy()
    for x_batch, y_batch in datagen.flow(x, y, batch_size=1, seed=1000):
        x2 = np.vstack([x2, x_batch])
        y2 = np.vstack([y2, y_batch])
        n = n - 1
        if n == 0:
            break
    return x2, y2


def csvLog(y, filename, row):
    filepath = 'csv/{}.csv'.format(filename)
    if not os.path.isfile(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(csvHeaders[y])
            writer.writerow(row)
    else:
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def adjustRange(r, nl, img_size):
    r[0] = int(r[0])
    r[1] = int(r[1])

    if r[0] <= 0:
        r[0] = 0
        r[1] = nl
    elif r[1] >= img_size:
        r[0] = img_size - nl
        r[1] = img_size
    return r


def getLargestSlices(y):
    ls = []
    for i in range(y.shape[0]):
        largestSlice = 0
        largestArea = 0
        for j in range(y.shape[1]):
            s = np.sum(y[i, j, :, :])
            if s > largestArea:
                largestSlice = j
                largestArea = s
        ls.append(largestSlice)

    return ls


def create_data(src, mask, label=0, resize=(img_size, img_size)):
    maskF = '**/*_' + mask + '.nii.gz'
    files = glob.glob(src + maskF, recursive=True)
    name = ''
    # print('Processing---', mask)
    imgs = []
    for i, file in enumerate(files):
        if (file.find('_seg') > 0 and label == 0):
            continue
        img = io.imread(file, plugin='simpleitk')
        if label == 1:
            img = 0
            if (file.find('HGG') > 0):
                img = 1
            imgs.append(img)
            continue
        elif label == 2:
            # img[img < 1] = 0
            img[img != 0] = 1  # Region 1 => 1+2+3+4 complete tumor
            img = img.astype('float32')
        elif label == 3:
            img = img.astype('float32')
        else:
            img = (img - img.mean()) / img.std()  # normalization => zero mean   !!!care for the std=0 problem
            # img = n4itk(img)

        # -----------------

        # for slice in range(50,130):
        #     img_t = img[slice,:,:]
        #     # if not label:
        #     #    img_t = n4itk(img_t)
        #     img_t = img_t.reshape((1,)+img_t.shape)   #become rank 4
        #     imgs.append(img_t)

        # -----------------
        img_tmp = img[50:130]
        img_tmp = np.array(img_tmp)
        imgs.append(img_tmp)
        # -----------------

    name = 'data/'
    name += 'x_2_' + mask if label == 0 else 'y_2_' + y_file_names[label - 1]
    name += '_' + str(img_size)
    np.save(name, np.array(imgs).astype('float32'))  # save at home
    print('Saved', len(files), 'to', name, 'shape', np.array(imgs).shape)


def create_new_data(src, mask, label=0, resize=(256, 256)):  # (W,H)
    maskF = '**/' + mask + '/*.png'
    files = glob.glob(src + maskF, recursive=True)

    name = ''
    # print('Processing---', mask)
    imgs = []
    for i, file in enumerate(files):
        img = io.imread(file)  # ,plugin='simpleitk'
        img_tmp = resize_slice(img, resize)
        if label:
            img_tmp = img_tmp / 255.0
        img_tmp = np.array(img_tmp)
        img_tmp = img_tmp.reshape((1,) + img_tmp.shape)
        imgs.append(img_tmp)
    print(np.array(imgs).shape)
    imgs = np.array(imgs)
    name = 'data/'
    name += 'x_CAD' if label == 0 else 'y_CAD'
    np.save(name, imgs)  # save at home
    print('Saved', len(files), 'to', name, 'shape', np.array(imgs).shape)


def create_masked_data(seq, nl):
    x_raw = np.load('data/x_{}_240.npy'.format(seq))
    y = np.load('data/y_complete_240.npy')
    x_name = 'data/x_{}_mask_240.npy'.format(seq)
    maxX = 0
    maxY = 0
    x = x_raw * y
    print("x : ", np.array(x).shape)
    np.save(x_name, np.array(x).astype('float32'))  # save at home


def create_masked_data_(seq, nl):
    x_raw = np.load('data/x_2_{}_240.npy'.format(seq))
    y = np.load('data/y_2_complete_240.npy')
    maxX = 0
    maxY = 0
    x_ = x_raw  # * y
    x_name = 'data/x_{}_{}.npy'.format(seq, nl)
    x_2_name = 'data/x_2_{}_{}.npy'.format(seq, nl)
    y_name = 'data/y_core_{}.npy'.format(nl)
    ls = getLargestSlices(y)

    x = []
    x2 = []
    # y2 = []
    for i in range(y.shape[0]):
        largestSlice = ls[i]

        ss = np.where(y[i, largestSlice] == 1)
        cx = 120
        cy = 120
        if ss[0].shape[0] != 0:
            cx = (max(ss[0]) + min(ss[0])) / 2
            cy = (max(ss[1]) + min(ss[1])) / 2
        xr = [cx - nl / 2, cx + nl / 2]
        yr = [cy - nl / 2, cy + nl / 2]

        xr = adjustRange(xr, nl, 240)
        yr = adjustRange(yr, nl, 240)
        x_tmp = []
        x2_tmp = []
        for j in range(y.shape[1]):
            img_x = x_[i, j, xr[0]:xr[1], yr[0]:yr[1]]
            # img_y = y_[i,j,xr[0]:xr[1],yr[0]:yr[1]]
            x_tmp = img_x.reshape((1,) + img_x.shape)
            x2_tmp.append(img_x)
            x.append(x_tmp)
            # y_tmp.append(img_y)
        x2.append(x2_tmp)
        # y2.append(y_tmp)
    print("x, x_2: ", np.array(x).shape, np.array(x2).shape)
    np.save(x_name, np.array(x).astype('float32'))  # save at home
    np.save(x_2_name, np.array(x2).astype('float32'))  # save at home
    # np.save(y_name, np.array(y2).astype('float32'))  # save at home


def create_big5_data():
    x_fn = get_x_file_names()
    y = np.load('data/y_2_complete_240.npy')
    ls = getLargestSlices(y)
    for i in range(4):
        x_ = np.load('data/x_2_{}_240.npy'.format(x_fn[i]))
        xt = []
        if i == 0:
            yt = []
        for j in range(x_.shape[0]):
            lss = ls[j] - 2
            if (lss < 0):
                lss = 0
            xt.append(x_[j, lss: lss + 5, :, :])
            if i == 0:
                yt.append(y[j, lss: lss + 5, :, :])
        xt = np.array(xt)
        if i == 0:
            yt = np.array(yt)
        xt = xt * yt
        print(i, xt.shape)
        if i == 0:
            x = xt
        else:
            x = np.concatenate([x, xt], axis=1)
    x_name = 'data/x_2_Big5_mask_240.npy'
    np.save(x_name, np.array(x).astype('float32'))  # save at home


# create_big5_data()
def split_test_set(x, y, n):
    l = x.shape[0]
    x_train = x[n:-n]
    y_train = y[n:-n]
    x_test = np.concatenate([x[:n], x[-n:]], axis=0)
    y_test = np.concatenate([y[:n], y[-n:]], axis=0)
    return x_train, x_test, y_train, y_test


def animate(pat, k, gifname):
    # def animate(gifname):
    fig = plt.figure()
    # def update(i):
    if 0:  # gifname == "Predict.gif" or gifname == "GroundTruth.gif" :
        p = np.full((832, 832), pat[i])
        anim = plt.imshow(p, cmap='gray')
    elif k == -1:
        anim = plt.imshow(pat[i, :, :].squeeze(), cmap='gray')
    else:
        anim = plt.imshow(pat[i, k, :, :], cmap='gray')
        # anim.paste(a,(anim.size[0], 0))
    return anim


# a = animation.FuncAnimation(fig, update, frames=range(len(pat)), interval=400, blit=True)
# a.save(gifname)
# print(gifname + "saved")

# catch BRATS2017 Data
# create_data('D:/WenWeiSexy/MICCAI_BraTS_2019_Data_Training/', 't1', label=0)
# create_data('D:/WenWeiSexy/MICCAI_BraTS_2019_Data_Training/', 't1ce', label=0)
# create_data('D:/WenWeiSexy/MICCAI_BraTS_2019_Data_Training/', 't2', label=0)
# create_data('D:/WenWeiSexy/MICCAI_BraTS_2019_Data_Training/', 'flair', label=0)
# create_data('D:/WenWeiSexy/MICCAI_BraTS_2019_Data_Training/', 'seg', label=1)    # grade
# create_data('D:/WenWeiSexy/MICCAI_BraTS_2019_Data_Training/', 'seg', label=2)    # complete
# create_data('D:/WenWeiSexy/MICCAI_BraTS_2019_Data_Training/', 'seg', label=3)    # core

"""

y = np.load('data/y_CAD.npy')

y2 = y[100,0,:,:]
print(y2.shape)

for k in range(256):
    p = max(y2[k])
    if p != 0:
        print(k,p)
"""

#step1: 解註解把image load進來
#會有一張失敗
link = 'C:/Users/owner/Downloads/CAD'
create_new_data(link, 'Ori', label=0)
create_new_data(link, 'mask', label=1)

# create_new_data('C:/Users/Happy Trololo/Desktop', 'HO_GE_KO', label=0)
# create_new_data('C:/Users/Happy Trololo/Desktop', 'HO_GE_KOMASS', label=1)

# nl = 144

# create_masked_data('t1', nl)
# create_masked_data('t1ce', nl)
# create_masked_data('t2', nl)
# create_masked_data('flair', nl)
# x = np.load('data/x_t2_240.npy')
# y = np.load('data/y_complete_240.npy')
# print("x, y: ", x.shape, y.shape)


# x_ = x * y
# name = 'data/x_3_t2_240.npy'
# np.save(name, np.array(x_).astype('uint8'))  # save at home
# x 29-216 (187)
# y 45-194 (149)
# max length : x137 y125

# y = np.load('data/y_complete_240.npy')


# for i in range(4):
#     x_ = np.load('data/x_2_{}_144.npy'.format(x_file_names[i]))
#     if i == 0:
#         x = x_
#     else:
#         x = np.concatenate([x, x_], axis=1)
# name = 'data/x_2_All_144.npy'
# np.save(name, x)  # save at home
# print('Saved', name, 'shape', x.shape)
# x2 = np.load('data/x_2_t2_144.npy')
# x3 = np.load('data/x_2_t2_mask_144.npy')
# y = np.load('data/y_core_144.npy')
# y = y[0,0]
# y[y==4] = 3
# print(y.shape)
# y_ = np_utils.to_categorical(y,4)
# print(y_.shape)
# y_2 = np.argmax(y_, axis = -1)
# print(y_2.shape)
# for i in range(4):
# ww = np.where(y == i)
# if i == 0:
#     a = sum(y[y == i]+1)
# else:
#     a = sum(y[y == i])
#     a /= i
# print(i, a)
# wx = ww[0]
# wy = ww[1]
# print(i, y[wx[0],wy[1]], y_[wx[0],wy[1]], y_2[wx[0],wy[1]])


# x= np.load('data/x_flair_240.npy')
# y = np.load('data/y_core_240.npy')
# y2 = np.load('data/y_core_240.npy')
# y_ = np.load('data/y_core_144.npy')
# for n in range(10):
#     i = int(r.random() * y.shape[0])
#     while y[i,:,:].sum() < 800: #y3[i,40,:,:].sum() ==  y4[i,40,:,:].sum():
#        i = int(r.random() * y.shape[0])
#     plt.figure(figsize=(15,10))

#     plt.subplot(131)
#     plt.title('T2FLAIR')
#     plt.imshow(x[i, 0, :, :],cmap='gray')

#     plt.subplot(132)
#     plt.title('Ground Truth')
#     plt.imshow(y[i, 0, :, :],cmap='gray')

#     plt.show()


# # for i in range(4):
#    x = np.load('data/x_2_{}_240.npy'.format(x_fn[i]))
#    x_ = x * y
#    name = 'data/x_3_{}_240.npy'.format(x_fn[i])
#    np.save(name, np.array(x_).astype('uint8'))  # save at home