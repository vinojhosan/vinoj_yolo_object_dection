import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from data_generator import BatchGenerator
import yolo_model
import yolo_loss

from config import IMAGE_H, IMAGE_W, TRUE_BOX_BUFFER, GRID_H, GRID_W, BOX, CLASS


early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint('weights_blood.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)


optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = yolo_model.model_creation(IMAGE_H, IMAGE_W, TRUE_BOX_BUFFER, GRID_H, GRID_W, BOX, CLASS)
model.compile(loss=yolo_loss.my_custom_loss, optimizer=optimizer)

batch_generator = BatchGenerator(r'data/full_dataset.csv',
                                 r'/home/vinoj/vinoj/keras-frcnn/JPEGImages/')

model.fit_generator(generator        = batch_generator,
                    steps_per_epoch  = len(batch_generator),
                    epochs           = 100,
                    verbose          = 1,
                    callbacks        = [early_stop, checkpoint, tensorboard])

model.save("trained_blood.h5")
