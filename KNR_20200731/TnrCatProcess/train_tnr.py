import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenet_v2
import numpy as np


img_size = 128

mode = 'bbs' # [bbs, lmks]
if mode is 'bbs':
  output_size = 4
elif mode is 'lmks':
  output_size = 18

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

data_00 = np.load('C:/Users/USER/Desktop/test/preprocess/CAT_00.npy', allow_pickle=True)
data_01 = np.load('C:/Users/USER/Desktop/test/preprocess/CAT_01.npy', allow_pickle=True)
data_02 = np.load('C:/Users/USER/Desktop/test/preprocess/CAT_02.npy', allow_pickle=True)
data_03 = np.load('C:/Users/USER/Desktop/test/preprocess/CAT_03.npy', allow_pickle=True)
# data_04 = np.load('C:/Users/USER/Desktop/test/preprocess/CAT_04.npy', allow_pickle=True)
# data_05 = np.load('C:/Users/USER/Desktop/test/preprocess/CAT_05.npy', allow_pickle=True)
# data_06 = np.load('C:/Users/USER/Desktop/test/preprocess/CAT_06.npy', allow_pickle=True)

tnr_data_00 = np.load('C:/Users/USER/Desktop/test/preprocess/TNRCAT_00.npy', allow_pickle=True)
tnr_data_01 = np.load('C:/Users/USER/Desktop/test/preprocess/TNRCAT_01.npy', allow_pickle=True)

x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs')), axis=0)
# y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)
y_train = tnr_data_00

x_test = np.array(data_03.item().get('imgs'))
# y_test = np.array(data_06.item().get(mode))
y_test = tnr_data_01


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = x_train.astype('float32') / 255.
y_test = x_test.astype('float32') / 255.
y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))

#
mobilenetv2_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=True, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

callback = [TensorBoard(log_dir='logs/%s' % (start_time)),ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')]

model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True,validation_data=(x_test, y_test), verbose=1,callbacks=callback)

