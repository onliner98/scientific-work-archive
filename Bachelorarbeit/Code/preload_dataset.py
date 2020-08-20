from PIL import Image as im
import numpy as np
import os
# Preload x_train, y_train, x_val, y_val, x_test, y_test
IMG_SIZE = 224
CLASSES = ['drill', 'hammer', 'pliers', 'saw', 'screwdriver', 'wrench']

x_train = np.array([])
y_train = np.array([])

for c in range(len(CLASSES)):
    _,_,files = next(os.walk(os.path.join('dataset/train',CLASSES[c])))
    num_files = len(files)
    print(f'{CLASSES[c]} contains {num_files} files')
    class_arr = np.array([[0,0,0,0,0,0]])
    class_arr[0][c] = 1
    print(f'class_arr for {CLASSES[c]} is {class_arr} with class index {c}')
    for file in files:
        img = im.open(os.path.join('dataset/train',CLASSES[c],file))
        img = img.resize((IMG_SIZE,IMG_SIZE))
        img = np.array(img)
        img = np.array([img])
        if x_train.shape == (0,):
            x_train = img
            y_train = class_arr
        else:
            x_train = np.concatenate((x_train,img), axis=0)
            y_train = np.concatenate((y_train,class_arr), axis=0)
    print(f'x_train shape is {x_train.shape}, y_train shape is y_train.shape')
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)

x_val = np.array([])
y_val = np.array([])

for c in range(len(CLASSES)):
    _,_,files = next(os.walk(os.path.join('dataset/val',CLASSES[c])))
    num_files = len(files)
    print(f'{CLASSES[c]} contains {num_files} files')
    class_arr = np.array([[0,0,0,0,0,0]])
    class_arr[0][c] = 1
    print(f'class_arr for {CLASSES[c]} is {class_arr} with class index {c}')
    for file in files:
        img = im.open(os.path.join('dataset/val',CLASSES[c],file))
        img = img.resize((IMG_SIZE,IMG_SIZE))
        img = np.array(img)
        img = np.array([img])
        if x_val.shape == (0,):
            x_val = img
            y_val = class_arr
        else:
            x_val = np.concatenate((x_val,img), axis=0)
            y_val = np.concatenate((y_val,class_arr), axis=0)
    print(f'x_val shape is {x_val.shape}, y_val shape is y_val.shape')
np.save('x_val.npy', x_val)
np.save('y_val.npy', y_val)


x_test = np.array([])
y_test = np.array([])

for c in range(len(CLASSES)):
    _,_,files = next(os.walk(os.path.join('dataset/test',CLASSES[c])))
    num_files = len(files)
    print(f'{CLASSES[c]} contains {num_files} files')
    class_arr = np.array([[0,0,0,0,0,0]])
    class_arr[0][c] = 1
    print(f'class_arr for {CLASSES[c]} is {class_arr} with class index {c}')
    for file in files:
        img = im.open(os.path.join('dataset/test',CLASSES[c],file))
        img = img.resize((IMG_SIZE,IMG_SIZE))
        img = np.array(img)
        img = np.array([img])
        if x_test.shape == (0,):
            x_test = img
            y_test = class_arr
        else:
            x_test = np.concatenate((x_test,img), axis=0)
            y_test = np.concatenate((y_test,class_arr), axis=0)
    print(f'x_test shape is {x_test.shape}, y_test shape is {y_test.shape}')
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)