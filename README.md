# UNet-Segmentation

Передо мной стояла следующая задача: Дана обучающая выборка, которая состоит из нескольких тысяч фотографий автомобилей. Необходимо обучить нейронную сеть отделять объект автомобиль от фона. 

## Задача сегментации
Сегментация – задача разбиения изображений на области, соответствующие различным объектам.

<img src="https://user-images.githubusercontent.com/34841826/181710381-e095745f-64d2-4ba9-94b6-7f8523fdff79.png" width="600" height="300">

Можно привести аналогию с задачей классификации, в задаче сегментации также производится классификация, только не объектов, а отдельных пикселей. И результатом решения такой задачи, будет не конкретный ответ: принадлежность классу, а изображение с пикселями, принадлежащими разным классам.

## Структура сети UNet

В данной работе за основу была взята архитектура UNet.
Данная сеть является свёрточной нейронной сетью, которая была создана в 2015 году для сегментации биомедицинских изображений в отделении Computer Science Фрайбургского университета. 
<img src="https://user-images.githubusercontent.com/34841826/181710861-5375154c-f3ed-46b0-9047-6bfd643bdc77.png" width="1000" height="500">
Архитектура сети представляет собой последовательность слоёв свёртка+пулинг, которые сначала уменьшают пространственное разрешение картинки, а потом увеличивают его, предварительно объединив с данными картинки и пропустив через другие слои свёртки. Таким образом, сеть выполняет роль своеобразного фильтра.

Так сеть содержит сжимающий путь (слева) и расширяющий путь (справа), поэтому архитектура похожа на букву U, что и отражено в названии. На каждом шаге мы удваиваем количество каналов признаков. 

## Обучение

Сеть обучается методом стохастического градиентного спуска на основе входных изображений и соответствующих им карт сегментации. Из-за сверток выходное изображение меньше входного сигнала на постоянную ширину границы. 
Кросс-энтропия, вычисляемая в каждой точке, определяется как

<img src="https://user-images.githubusercontent.com/34841826/181711474-de2dad1b-6083-4fe4-a72f-486c6bafbb9f.png" width="250" height="90">
Граница разделения вычисляется с использованием морфологических операций.

## Загрузка и подготовка данных

Для обучения модели использовался dataset с различными изображениями и файл с бинарными масками.

<img src="https://user-images.githubusercontent.com/34841826/181715148-727750f2-be83-4a7e-ae2d-4f528fee44f5.png" width="600" height="400">
 
И маски для данных картинок

<img src="https://user-images.githubusercontent.com/34841826/181718109-bc4b8421-797c-49a8-85ce-9f3ba19e6a8c.png" width="600" height="150">

Код для импорта данных: 

    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
    !cp /content/gdrive/'My Drive'/data.zip .
    !unzip data.zip

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import pandas as pd

Более подробно в Application 1 - Import data.

## Создание архитектуры

Сначала выполнялась свертка функцией Conv2D с параметром padding, который сохранял пограничные пиксели. После этого выполнялась активация (Activation с параметром Relu).  

    inp = Input(shape=(256, 256, 3)) 

    conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp) # слой свертки
    conv_1_1 = Activation('relu')(conv_1_1) # активация свертки

    conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
    conv_1_2 = Activation('relu')(conv_1_2)

    pool_1 = MaxPooling2D(2)(conv_1_2) # пулинг
    

Во второй половине обучения происходит увеличение изображения и параллельная конкатенация блоков одинаковой размерности.
Здесь применяются функции UpSampling2D с билинейной интерполяцией для увеличения изображения. Также реализуется блок из 2х сверток с их последующей активацией.	

    up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4) # апсамплинг 
    conc_1 = Concatenate()([conv_4_2, up_1]) # конкатенация с тензором пулинга той же размерности

    conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(conc_1) # свертка уже сконкатенированного блока
    conv_up_1_1 = Activation('relu')(conv_up_1_1)

    conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
    conv_up_1_2 = Activation('relu')(conv_up_1_2)
    

На последнем этапе происходит свертка изображения с 1м каналом, для определения конечной маски и активация с параметром Sigmoid для получения вероятностей пикселей маски.

    conv_up_4_2 = Conv2D(1, (3, 3), padding='same')(conv_up_4_1) # свертка с 1м каналом, для вывода маски изображения
    result = Activation('sigmoid')(conv_up_4_2)

Более подробно в Application 2 - Library import.

## Обучение модели

На данном этапе создается модель функцией Model.

    model = Model(inputs=inp, outputs=result)
    
Компиляция модели, в качестве функции ошибки берем кросс-энтропию.

    model.compile(adam, 'binary_crossentropy')

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
    
Процесс обучения модели:

<img src="https://user-images.githubusercontent.com/34841826/181714876-a0fe6dff-cb42-45f8-bf72-1bf10e00cb7c.png" width="600" height="400">

Более подробно в Application 3 - Running Model Training.

## Полученные результаты работы нейронной сети

Вот некоторые из полученных данных сегментации. Нашей задачей было выделить маску машины по изображению (т.е. отделить объект машина от фона). И вот как это реализовала наша нейронная сеть:

<img src="https://user-images.githubusercontent.com/34841826/181715062-3db695d0-7f78-40d7-8a06-b0eedfe2a04d.png" width="750" height="300">

<img src="https://user-images.githubusercontent.com/34841826/181715093-bb05956e-0f48-430e-af69-b1c70f7f4419.png" width="750" height="300">


