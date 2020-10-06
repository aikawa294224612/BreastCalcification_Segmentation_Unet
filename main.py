# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:49:46 2019

@author: wenwe
"""
#from AUnet import *
from globalSetting import *
from preprocess import *
from my_models import *
from keras.callbacks import CSVLogger
from sklearn.utils import shuffle
from importlib import reload
import time

# import visualization_utils  as vu

x_fn = get_x_file_names()
y_fn = get_y_file_names()
img_size = get_img_size()
batch_size = get_batch_size()
num_of_aug = get_num_of_aug()
num_epoch = get_num_epoch()  # 50
aug_num = get_aug_num()
K.tensorflow_backend._get_available_gpus()
# K.set_image_data_format("channels_first")
# K.set_image_dim_ordering("th")
# setImg_size(img_size)
# %%


# %%
# load numpy array data

xi = 3  # ['t1', 't1ce', 't2', 'flair', 'All', 'Big5']
yi = 1  # ['grade', 'complete', 'core']

aug = 1

use_mask = False
if use_mask:
    msk = "_mask"
else:
    msk = ""
# model = unet_model2()
# k = 0
# %%
csv_logger = CSVLogger('log/training_' + y_fn[yi] + '.log', separator=',', append=False)
# training

for xx in range(1):  # if 1:
    # reload(my_models)
    if 1:
        x = np.load('data/x_CAD.npy')
        y = np.load('data/y_CAD.npy')
        # x, y, aug_method = augmentation_methods(x, y, aug)

    elif yi == 0:
        xi = 4 + xx
        x = np.load('data/x_2_{}_{}.npy'.format(x_fn[xi], img_size))
        y = np.load('data/y_{}.npy'.format(y_fn[yi]))
        # x, y = augmentation_4_classification(x,y,aug_num)
    else:
        x = np.load('data/x_{}_{}.npy'.format(x_fn[xi] + msk, img_size))
        y = np.load('data/y_{}_{}.npy'.format(y_fn[yi], img_size))
        aug_method = "none"
        # x, y, aug_method = augmentation_methods(x, y, aug)
    # csvName = '{}/{}'.format(y_fn[yi], x_fn[xi])
    # csvName = '{}'.format(y_fn[yi])
    csvName = "CAD"
    # csvLog(csvName, ['aug : {}'.format(aug_num[j])])
    x, y = shuffle(x, y, random_state=0)
    # x = x[:1000]
    # y = y[:1000]
    if yi == 2:
        y[y == 4] = 3
        y = np_utils.to_categorical(y, 4)
        y = y[:, 0, :, :, :]
    print("x, y: ", x.shape, y.shape)
    # for ccc in range(10):
    #    model = cnn_model(x.shape[1])
    #    cw = (ccc + 1) * 0.1
    #    class_weight = {0: 1,
    #                    1: cw}
    #    datagen = data_generator()

    #    csv_loggerName = 'log/training_{}_{}_{}_{}_{}_{}.log'.format(x_fn[xi], y_fn[yi], img_size, aug_num, num_epoch, cw)
    #    csv_logger = CSVLogger(csv_loggerName, separator=',', append=True)
    #    history = model.fit(x, y, batch_size=batch_size, validation_split=0.1 ,nb_epoch= num_epoch, verbose=1,
    #                         shuffle=True, callbacks=[csv_logger], class_weight = class_weight)
    #    model.save_weights('weights/weights_{}_{}_{}_{}_{}_{}.h5'.format(x_fn[xi], y_fn[yi], img_size, aug_num, num_epoch, cw))
    #    #model.load_weights('weights_{}_{}_{}.h5'.format(img_size, num_epoch, y_fn[yi]))
    #    pred = model.predict(x)
    #    #print(history.history.keys())
    #    acc = my_accuracy(y,pred)
    #    acc = '{:.2f}'.format(acc)
    #    val = max(history.history['val_binary_accuracy']) * 100
    #    val = '{:.2f}'.format(val)
    #    #acc = max(history.history['my_accuracy_metric']) * 100
    #    #acc = '{:.2f}'.format(acc)
    #    sen = max(history.history['sensitivity']) * 100
    #    sen = '{:.2f}'.format(sen)
    #    spec = max(history.history['specificity']) * 100
    #    spec = '{:.2f}'.format(spec)
    #    #['img_size', 'epoch', 'class weight', 'accuracy', 'my acc', 'sen', 'spec']
    #    logger = [img_size, num_epoch, '1:{:.1f}'.format(cw), val, acc, sen, spec]
    #    csvLog(yi, csvName, logger)
    #    #print("Hit {} in {} cases. Accuracy {}".format(ya, x.shape[0], acc))
    #    del model, pred, history
    #    K.clear_session()
    # del x, y
    # print("predict {} as {}".format(y[i], pred[i]))

    # ------------------

    if 1 or yi == 1:
        model = unet_latest()
        #model = unet_model()  # aunet()
    else:
        model = unet_model2()
    history = model.fit(x, y, batch_size=batch_size, validation_split=0.2, epochs=num_epoch, verbose=1,
                        shuffle=True, callbacks=[csv_logger])
    modelName = '256x50eAUNET.h5'
    model.save_weights('weights/' + modelName)
    print(modelName)
    # model.save_weights('weights/weights_{}_{}_{}_{}_{}.h5'.format(x_fn[xi] + msk, y_fn[yi], img_size, aug_num, num_epoch))
    # model.load_weights('weights/weights_{}_{}_{}_{}_{}.h5'.format(x_fn[xi]+msk , y_fn[yi], img_size, aug_num, num_epoch))
    # model.load_weights('weights_CAD_.h5')

    if 1 or yi == 1:
        Dice = max(history.history['val_dice_coef']) * 100
        Dice = '{:.2f}'.format(Dice)
        Acc = max(history.history['val_acc']) * 100
        Acc = '{:.2f}'.format(Acc)
        logger = ['CAD', img_size, num_epoch, Dice, Acc, '    ' + modelName]
        # logger = [x_fn[xi], aug_method, img_size, num_epoch, Dice, Acc]
    else:
        Dice = max(history.history['val_dice_coef_multi']) * 100
        Dice = '{:.2f}'.format(Dice)
        Dice_ET = max(history.history['val_dice_coef_ET']) * 100
        Dice_ET = '{:.2f}'.format(Dice_ET)
        Dice_ED = max(history.history['val_dice_coef_ED']) * 100
        Dice_ED = '{:.2f}'.format(Dice_ED)
        Dice_Core = max(history.history['val_dice_coef_Core']) * 100
        Dice_Core = '{:.2f}'.format(Dice_Core)
        logger = [x_fn[xi], use_mask, img_size, num_epoch, Dice, Dice_ET, Dice_ED, Dice_Core]
    csvLog(yi, csvName, logger)

"""
rSeq = []
for i in range(30):
    n = int(r.random() * y.shape[0])
    #while y[n,:,:,2:].sum() < 100 or y[n,:,:,1:].sum() < 500 or n in rSeq:
    while y[n,:,:].sum() < 100 or n in rSeq:
       n = int(r.random() * y.shape[0])
    rSeq.append(n)

x_ = []
y_ = []

for n in rSeq:
   x_.append(x[n])
   y_.append(y[n])

for i in range(1):
   x_.append(x[i])
   y_.append(y[i])

x_ = np.array(x_)
y_ = np.array(y_)

print("x2, y2: ", x_.shape, y_.shape)


pred = model.predict(x_)

imgDir = 'img/832x100e_0901_1/'
#imgDir = 'img/{}_{}_{}/'.format(x_fn[xi], y_fn[yi], img_size)

for i in range(1):
   plt.imshow(pred[i,:, :].squeeze(),cmap='gray') #Needs to be in row,col order
   plt.savefig(imgDir+'Predic.png')

print('saved>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

if not os.path.isdir(imgDir):
   os.mkdir(imgDir)

# for i in range(4):

#    s = pred[:,:,:,i]
#    print(i, s.sum())

if yi == 2:
   pred = np.argmax(pred, axis = -1)
   y_ = np.argmax(y_, axis = -1)
print(pred.shape)


# for j in range(1):
#    for i in range(2):
#       n = i * 3 + 1
#       k = i + 2 * j
#       plt.subplot(2,3,n)
#       if i == 0:
#          plt.title('(a)')
#       plt.imshow(x_[k, 0, :, :],cmap='gray')

#       plt.subplot(2,3,n + 1)
#       if i == 0:
#          plt.title('(b)')
#       plt.imshow(y_[k, :, :],cmap='gray')

#       plt.subplot(2,3,n + 2)
#       if i == 0:
#          plt.title('(c)')
#       plt.imshow(pred[k, :, :],cmap='gray')

#    plt.show()

imgDir = 'img/CAD_test'
#imgDir = 'img/{}_{}_{}/'.format(x_fn[xi], y_fn[yi], img_size)

if not os.path.isdir(imgDir):
   os.mkdir(imgDir)


#animate(pred, -1, imgDir + 'Predict.png')
#animate(x_, 0, imgDir + '{}.png'.format(x_fn[xi]))
#animate(y_, -1, imgDir + 'GroundTruth.png')

# animate(x_, 1, 'T1ce.gif')
# animate(x_, 2, 'T2.gif')
# animate(x_, 3, 'Flair.gif')

# summarize history for accuracy
#plt.plot(history.history['dice_coef'])
#plt.plot(history.history['val_dice_coef'])

plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""

# t_sample = datagen.flow(x, y, batch_size=batch_size, subset="training")
# v_sample = datagen.flow(x, y, batch_size=batch_size, subset="validation")
# v_step = int(len(t_sample)/batch_size)
# print("step:", len(t_sample), len(v_sample), v_step)

# history = model.fit_generator(t_sample, nb_epoch= num_epoch,
#                               validation_data = v_sample, validation_steps = v_step,
#                               steps_per_epoch=x.shape[0],
#                               verbose=1, shuffle=True, callbacks=[csv_logger], class_weight = class_weight)


# %%
