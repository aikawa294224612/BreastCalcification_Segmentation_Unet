def get_img_size():
    return img_size


def get_num_of_aug():
    return num_of_aug


def get_num_epoch():
    return num_epoch


def get_x_file_names():
    return x_file_names


def get_y_file_names():
    return y_file_names


def get_aug_num():
    return aug_num


def get_aug_methods():
    return aug_methods


def get_batch_size():
    return batch_size


def get_csvHeaders():
    return csvHeaders


img_size = 256  # original img size is 240*240
num_epoch = 50
num_of_aug = 1
x_file_names = ['t1', 't1ce', 't2', 'flair', 'All', 'Big5']
y_file_names = ['grade', 'complete', 'core']
aug_methods = ['none', 'random', 'bigBros']
aug_num = 335  # 3350
batch_size = 1

csvHeaders = [['img_size', 'epoch', 'class weight', 'accuracy', 'my acc', 'sen', 'spec'],
              ['Sequence', 'augmentation', 'img_size', 'num_epoch', 'Dice', 'Acc'],
              ['Sequence', 'use_mask', 'img_size', 'num_epoch', 'Dice', 'Dice_ET', 'Dice_ED', 'Acc']
              ]