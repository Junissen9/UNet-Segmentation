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