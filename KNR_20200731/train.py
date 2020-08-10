import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.applications import mobilenet_v2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


base_path = 'C:/Users/server/Documents/ProjectTnr/dataset'


# input image size & output size
img_size = 224
output_size = 1

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


# load dataset
data_00 = np.load(base_path + '/CAT_00.npy', allow_pickle=True)
data_01 = np.load(base_path + '/CAT_01.npy', allow_pickle=True)
data_02 = np.load(base_path + '/CAT_02.npy', allow_pickle=True)
data_03 = np.load(base_path + '/CAT_03.npy', allow_pickle=True)
data_04 = np.load(base_path + '/CAT_04.npy', allow_pickle=True)
data_05 = np.load(base_path + '/CAT_05.npy', allow_pickle=True) # validationset

print('dataset load 완료')



# train set
x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs')), axis=0)
y_train = np.concatenate((data_00.item().get('lbs'), data_01.item().get('lbs'), data_02.item().get('lbs'), data_03.item().get('lbs')), axis=0)


# validation set
x_val = np.concatenate((data_04.item().get('imgs')), axis=0)
y_val = data_04.item().get('lbs')

# test set
x_test = np.concatenate((data_05.item().get('imgs')), axis=0)
y_test = data_05.item().get('lbs')

# 정규화
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# reshape
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_val = np.reshape(x_val, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))
y_train = np.reshape(y_train, (-1, 1))
y_val = np.reshape(y_val, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))


print('정규화 완료')


inputs = Input(shape=(img_size, img_size, 3))

mobilenetv2_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=True, weights=None, input_tensor=inputs, classes=2)

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='sigmoid')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()



# training



model.compile(optimizer=keras.optimizers.Adam(), metrics=['accuracy'], loss='mse')


history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_val, y_val),
                    verbose=1
                )




# 모델 평가
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ' + str(loss_and_metrics))


# 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 학습 손실 값과 검증 손실 값을 플롯팅 합니다.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# save model
model.save('C:/Users/server/Documents/ProjectTnr/model/' + str(start_time) + '.h5')

print('save 완료')
