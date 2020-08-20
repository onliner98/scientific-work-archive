import os
from keras import backend, optimizers, models
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import efficientnet

# Adept Constants, Model, ModelCheckpoint Filepath, optimizer params, fit params, early stopping
IMG_SIZE = 600
BATCH_SIZE = 1
INITIAL_EPOCH = 0
EPOCHS = 100
CLASSES = ['drill','hammer','pliers','saw','screwdriver','wrench']

# Create outputdir
os.makedirs('model_weights/efficientnet', exist_ok=True)

# Load data
assert(backend.image_data_format()=='channels_last')
train_batches = ImageDataGenerator().flow_from_directory(directory='dataset/train', target_size=(IMG_SIZE,IMG_SIZE), classes=CLASSES, batch_size=BATCH_SIZE)
val_batches = ImageDataGenerator().flow_from_directory(directory='dataset/val', target_size=(IMG_SIZE,IMG_SIZE), classes=CLASSES, batch_size=BATCH_SIZE)
input_shape = (IMG_SIZE,IMG_SIZE,3)

# Train model
model = efficientnet.EfficientNetB7(input_shape, 6)
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.256, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
if INITIAL_EPOCH != 0:
	model.load_weights('latest-efficientnet.hdf5')
	print('loaded weights successfully')

mc = ModelCheckpoint('model_weights/efficientnet/latest-efficientnet.hdf5', monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
mcb = ModelCheckpoint('model_weights/efficientnet/best-efficientnet.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
def schedule(epoch, current_lr):
    for k in range(int(EPOCHS/2.4)):
        k = k+1
        target_epoch=round(k*2.4, 0)
        if target_epoch==epoch:
            return current_lr*0.97
    return current_lr
lrs = LearningRateScheduler(schedule, verbose=1)
steps_per_epoch = len(train_batches)
val_steps = len(val_batches)
log = CSVLogger('model_weights/efficientnet/efficientnet.log')
model.fit_generator(generator=train_batches, steps_per_epoch=steps_per_epoch, validation_data=val_batches, validation_steps=val_steps, epochs=EPOCHS, callbacks = [mc, mcb, lrs, log],initial_epoch=INITIAL_EPOCH)





# Load Data
test_batches = ImageDataGenerator().flow_from_directory(directory='dataset/test', target_size=(IMG_SIZE,IMG_SIZE), classes=CLASSES, batch_size=BATCH_SIZE)

# Load Model
model = efficientnet.EfficientNetB7(input_shape, 6)
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.256, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('model_weights/efficientnet/best-efficientnet.hdf5')
print('loaded weights successfully')

# Eval
out = model.evaluate_generator(test_batches, verbose=1)

# Save Results
f= open("results.txt","a+")
f.write(f"EfficientNetB7 test_loss: {out[0]}, test_accuracy:  {out[1]}\n")
print(f"wrote \"EfficientNetB7 test_loss: {out[0]}, test_accuracy:  {out[1]}\n\" to results.txt ")
f.close()