# Загрузка тренировочных данных
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

!cp /content/gdrive/'My Drive'/data.zip .
!unzip data.zip

# Импортирование библиотек
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Функция декодирования масок из последовательности 0 и 1
def rle_decode(mask_rle, shape=(1280, 1918, 1)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    img = img.reshape(shape)
    return img

# Считывание данных
df = pd.read_csv('data/train_masks.csv')

train_df = df[:4000]
val_df = df[4000:]

img_name, mask_rle = train_df.iloc[4]

img = cv2.imread('data/train/{}'.format(img_name))
mask = rle_decode(mask_rle)


# Генерация batch-пакетов нейронной сети
def keras_generator(gen_df, batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0] # считывание случайной картинки и маски
            img = cv2.imread('data/train/{}'.format(img_name))
            mask = rle_decode(mask_rle)
            
            img = cv2.resize(img, (256, 256)) 
            mask = cv2.resize(mask, (256, 256))
            
            x_batch += [img] # добавляем в batch
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255. # нормирование размера
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)

# Запуск генератора для 16-ти картинок
for x, y in keras_generator(train_df, 16): 
    break