import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
import keras as k
import ssd_simple_generator as dataset
import ssd_loss

input_h = 300
input_w = 300

grid_h = 10
grid_w = 10
batch_size = 8

ds = dataset.SimpleGenerator((input_h, input_w), (grid_h, grid_w), batch_size=8, n_images=1000)
n_class = ds.n_class

def get_model():
    input_layer = k.Input((input_h, input_w, 3), name='input_image')

    resnet50 = k.applications.ResNet50(include_top=False, input_tensor=input_layer, input_shape=(input_h, input_w, 3), weights='imagenet', pooling=None) #(input_layer)
    # resnet50.summary()
    # print(resnet50.shape)
    activation_49 = resnet50.get_layer('activation_49').output

    output_channel = n_class * 3
    resnet_out = k.layers.Conv2D(output_channel, (3, 3), padding='same', activation='relu')(activation_49)
    ssd_model = k.models.Model(input_layer, resnet_out)

    ssd_model.summary()

    return ssd_model

def identity_metric(y_true, y_pred):
    return k.backend.mean(y_true-y_pred)


def my_loss(y_true, y_pred):

    true_conf = y_true[:, :, :, 0:n_class]
    pred_conf = y_pred[:, :, :, 0:n_class]

    true_offset = y_true[:, :, :, n_class:]
    pred_offset = y_pred[:, :, :, n_class:]
    conf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_conf, logits=pred_conf))

    # Find loss for only that valid specific grid location [ignore non point offsets]
    true_offset_mask = tf.cast(true_offset > 0.1, tf.float32)  # Check if positive, then in that grid box
    offset_pred_at_valid_location = true_offset_mask * pred_offset

    offset_loss = tf.reduce_mean(tf.square(true_offset - offset_pred_at_valid_location))

    total_loss = conf_loss * 4 + offset_loss * 1
    return total_loss


def train():

    yolo_model = get_model()
    # yolo_model.load_weights('models/trained_model_32x32_20kp_batdataset.hdf5')

    adam = k.optimizers.Adam(lr=0.005)

    yolo_model.compile(adam, loss=my_loss, metrics=[identity_metric])

    # Model saving and checkpoint callbacks
    yolo_model.save('models/empty_model_10x10.hdf5')
    filepath = "models/weights.best_pose_10x10.hdf5"
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    tensorboard = k.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    # pose_aug_generator - Augments and reads images.  Obtains num_keypoints+rgb channels
    yolo_model.fit_generator(ds,
                             steps_per_epoch=ds.n_images // batch_size,
                             epochs=100,
                             callbacks=callbacks_list)

    yolo_model.save('models/trained_model_10x10.hdf5')


train()