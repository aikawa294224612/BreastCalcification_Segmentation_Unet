from globalSetting import *
from preprocess import *
from my_models import *
from keras.callbacks import CSVLogger
from sklearn.utils import shuffle
from importlib import reload
from PIL import Image
import os
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io

Sky = [128, 128, 128]
Building = [255, 255, 255]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

# import visualization_utils  as vu

x_fn = get_x_file_names()
y_fn = get_y_file_names()
img_size = get_img_size()
batch_size = get_batch_size()
num_of_aug = get_num_of_aug()
num_epoch = get_num_epoch()  # 20
aug_num = get_aug_num()
K.tensorflow_backend._get_available_gpus()
"""
#印出照片轉移
#path = 'C:/Users/Happy Trololo/Desktop/Kodak/Kodak/mass' #資料夾目錄
#files= os.listdir(path) #得到資料夾下的所有檔名稱
pathsave ='C:/Users/Happy Trololo/Desktop/KO/'
pathori ='C:/Users/Happy Trololo/Desktop/GEMass'
filesori= os.listdir(pathori) #得到資料夾下的所有檔名稱

pathtarget ='C:/Users/Happy Trololo/Desktop/GEMASSLABEL'
filestarget= os.listdir(pathtarget) #得到資料夾下的所有檔名稱

for i in range(len(filestarget)):
   for j in range(len(filesori)):
      if filestarget[i]==filesori[j]:
         ko = Image.open(pathori+'/'+str(filesori[j])); #開啟檔案
         ko.save(pathsave+str(filesori[j]))
   #path2= path+'/'+str(files[i])+'/mass'
   #files2= os.listdir(path2) #得到資料夾下的所有檔名稱
   #for j in range(len(files2)):
    #  ko = Image.open(path2+'/'+str(files2[j])); #開啟檔案
    #  ko.save(pathsave+'/'+str(files[i])+'_'+str(files2[j]))

    rcc = Image.open(path2+'/'+'rcc.png'); #開啟檔案
    rcc.save(pathsave+str(files[i])+'_rcc.png')

    rmlo = Image.open(path2+'/'+'rmlo.png'); #開啟檔案
    rmlo.save(pathsave+str(files[i])+'_rmlo.png')

    lcc = Image.open(path2+'/'+'lcc.png'); #開啟檔案
    lcc.save(pathsave+str(files[i])+'_lcc.png')

    lmlo = Image.open(path2+'/'+'lmlo.png'); #開啟檔案
    lmlo.save(pathsave+str(files[i])+'_lmlo.png')

print('Finish>>>>>>>>>>>>>>>>>>>>')
"""
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255.0


# 測試model
create_new_data('C:/Users/Happy Trololo/Desktop/', 'X', label=0)
# create_new_data('D:/WenWeiSexy/CAD', 'Ori', label=0)

model = unet_model()  # unet_model()
model.load_weights('weights/256x50eHGK.h5')

path = 'C:/Users/Happy Trololo/Desktop/X'  # 資料夾目錄
# path='D:/WenWeiSexy/CAD/Ori'
files = os.listdir(path)  # 得到資料夾下的所有檔名稱
x = np.load('data/x_CAD.npy')
x_ = []

for i in range(5):
    x_.append(x[i])
x_ = np.array(x_)
pred = model.predict(x_)
imgDir = 'C:/Users/Happy Trololo/Desktop/XTEST/'
for i in range(5):
    img_ad = pred[i, :, :].squeeze()
    img = labelVisualize(2, COLOR_DICT, img_ad)
    io.imsave(imgDir + str(files[i]) + 'HGK_Predic.png', img)

    # plt.imshow(img_ad,cmap='gray') #Needs to be in row,col order
    # plt.savefig(imgDir+str(files[i])+'Aug_Predic.png')
print('saved>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

"""

x2 = pred[i,:, :].squeeze()*10
   print(x2.shape)

   for k in range(832):
    p = max(x2[k])
    if p != 0:
       print(k,p)

"""