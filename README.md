# UNet-Segmentation
Image segmentation with neural networks
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

# Импортирование библиотек - в работе используется фреймворк Keras
import keras
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D, Activation
from keras.models import Model
from keras.layers import Input, Dense, Concatenate

inp = Input(shape=(256, 256, 3)) 

conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp) # слой свертки
conv_1_1 = Activation('relu')(conv_1_1) # активация свертки
conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
conv_1_2 = Activation('relu')(conv_1_2)
pool_1 = MaxPooling2D(2)(conv_1_2) # пулинг


conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
conv_2_1 = Activation('relu')(conv_2_1)
conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
conv_2_2 = Activation('relu')(conv_2_2)
pool_2 = MaxPooling2D(2)(conv_2_2)


conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
conv_3_1 = Activation('relu')(conv_3_1)
conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
conv_3_2 = Activation('relu')(conv_3_2)
pool_3 = MaxPooling2D(2)(conv_3_2)


conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
conv_4_1 = Activation('relu')(conv_4_1)
conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
conv_4_2 = Activation('relu')(conv_4_2)
pool_4 = MaxPooling2D(2)(conv_4_2)

# обратный блок
up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4) # апсамплинг 
conc_1 = Concatenate()([conv_4_2, up_1]) # конкатенация с тензором пулинга той же размерности

conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(conc_1) # свертка уже сконкатенированного блока
conv_up_1_1 = Activation('relu')(conv_up_1_1)
conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
conv_up_1_2 = Activation('relu')(conv_up_1_2)


up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
conc_2 = Concatenate()([conv_3_2, up_2])
conv_up_2_1 = Conv2D(128, (3, 3), padding='same')(conc_2)
conv_up_2_1 = Activation('relu')(conv_up_2_1)
conv_up_2_2 = Conv2D(128, (3, 3), padding='same')(conv_up_2_1)
conv_up_2_2 = Activation('relu')(conv_up_2_2)


up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
conc_3 = Concatenate()([conv_2_2, up_3])
conv_up_3_1 = Conv2D(64, (3, 3), padding='same')(conc_3)
conv_up_3_1 = Activation('relu')(conv_up_3_1)
conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
conv_up_3_2 = Activation('relu')(conv_up_3_2)


up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2) # последний апсемплинг
conc_4 = Concatenate()([conv_1_2, up_4])
conv_up_4_1 = Conv2D(32, (3, 3), padding='same')(conc_4)
conv_up_4_1 = Activation('relu')(conv_up_4_1)
conv_up_4_2 = Conv2D(1, (3, 3), padding='same')(conv_up_4_1) # свертка с 1м каналом, для вывода маски изображения
result = Activation('sigmoid')(conv_up_4_2)

# Запуск модели
model = Model(inputs=inp, outputs=result)

# Создание списков весов моделей 
best_w = keras.callbacks.ModelCheckpoint('unet_best.h5', # сохранение лучших весов модели
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

last_w = keras.callbacks.ModelCheckpoint('unet_last.h5', # сохранение последних весов модели
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=False,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

callbacks = [best_w, last_w]

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # оптимизация

# Компиляция модели, в качестве функции ошибки берем кросс-энтропию
model.compile(adam, 'binary_crossentropy')

# Собственно, запуск обучения модели
batch_size = 16
model.fit_generator(keras_generator(train_df, batch_size),
              steps_per_epoch=100, # кол-во батчей
              epochs=100, # кол-во эпох обучения 
              verbose=1, # вывод результата
              callbacks=callbacks, # наши листы весов
              validation_data=keras_generator(val_df, batch_size), # отдельный генератор для валидационной выборки
              validation_steps=50,
              class_weight=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              shuffle=True,
              initial_epoch=0)

pred = model.predict(x)# функция прогнозирования маски

im_id = 5
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
axes[0].imshow(x[im_id]) # вывод картинки
axes[1].imshow(pred[im_id, ..., 0] > 0.5) # предсказанной маски для нее

plt.show()





