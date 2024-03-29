import os
import numpy as np
from keras import backend, optimizers, models
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger

import resnext

# Adept Constants, Model, ModelCheckpoint Filepath, optimizer params, fit params, early stopping
IMG_SIZE = 224
BATCH_SIZE = 16
INITIAL_EPOCH = 0

# Create outputdir
os.makedirs('model_weights/resnext', exist_ok=True)

# Load data
assert(backend.image_data_format()=='channels_last')
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')
input_shape = (IMG_SIZE,IMG_SIZE,3)

# DEBUG set epoch 1 when using this code
#x_train = x_train[0:64]
#y_train = y_train[0:64]
#x_val = x_val[0:32]
#y_val = y_val[0:32]

print(f'x_train has shape {x_train.shape}')
print(f'y_train has shape {y_train.shape}')
print(f'x_val has shape {x_val.shape}')
print(f'y_val has shape {y_val.shape}')
print(f'input shape is {input_shape}')



# Train model
model = resnext.ResNeXt101(input_shape, 6)
model.compile(optimizer=optimizers.SGD(learning_rate=0.1, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
if INITIAL_EPOCH != 0:
	model.load_weights('model_weights/resnext/latest-resnext.hdf5')
	print('loaded weights successfully')

mc = ModelCheckpoint('model_weights/resnext/latest-resnext.hdf5', monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
mcb = ModelCheckpoint('model_weights/resnext/best-resnext.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
def schedule(epoch, lr):
  if epoch == 30 or epoch == 60 or epoch == 90:
    return lr / 10
  return lr
lrs = LearningRateScheduler(schedule)
log = CSVLogger('model_weights/resnext/resnext.log')

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=120, callbacks = [mc, mcb, log, lrs], validation_data=(x_val, y_val), shuffle=True, initial_epoch=INITIAL_EPOCH)





# Load Data
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Load Model
model = resnext.ResNeXt101(input_shape, 6)
model.compile(optimizer=optimizers.SGD(learning_rate=0.1, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('model_weights/resnext/best-resnext.hdf5')
print('loaded weights successfully')

# Eval
out = model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE, verbose=1)

# Save Results
f= open("results.txt","a+")
f.write(f"ResNeXt101 test_loss: {out[0]}, test_accuracy:  {out[1]}\n")
print(f"wrote \"ResNeXt101 test_loss: {out[0]}, test_accuracy:  {out[1]}\n\" to results.txt ")
f.close()