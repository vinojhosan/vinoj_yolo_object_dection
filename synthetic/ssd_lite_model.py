import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.layers import Conv2D, Reshape, Activation, Lambda
import synthetic_dataset as dataset

input_h = 224
input_w = 224

grid_h = 7
grid_w = 7
batch_size = 8

model_out_path = './models'
os.makedirs(model_out_path, exist_ok=True)
ds = dataset.SynthenticGenerator((input_h, input_w), (grid_h, grid_w),
                             batch_size=batch_size, n_images=10000)

val_ds = dataset.SynthenticGenerator((input_h, input_w), (grid_h, grid_w),
                             batch_size=batch_size, n_images=1000)

def get_model():
    input_layer = k.Input((input_h, input_w, 3), name='input_image')

    mobile_net = k.applications.MobileNetV2(include_top=False,
                                       weights='imagenet', pooling=None) #(input_layer)
    mobile_net_feature = mobile_net(input_layer)

    classes = Conv2D(ds.n_class+1, (3, 3), strides=(1, 1),
                      padding="same", kernel_initializer='he_normal', name='classes')(mobile_net_feature)

    boxes = Conv2D(4, (3, 3), strides=(1, 1), padding="same",
                             kernel_initializer='he_normal', name='boxes')(mobile_net_feature)

    classes_reshaped = Reshape((-1, ds.n_class+1), name='classes_reshape')(classes)
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_reshaped)

    boxes_reshaped = Reshape((-1, 4), name='boxes_reshape')(boxes)

    output = k.layers.Concatenate(axis=-1)([boxes_reshaped, classes_softmax])

    ssd_model = k.models.Model(input_layer, output)

    ssd_model.summary()

    return ssd_model


def ssd_loss(y_true, y_pred):

    def smooth_L1_loss(y_true, y_pred):
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(y_true, y_pred):

        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    # batch_size = tf.to_float(tf.shape(y_pred)[0])
    true_classes = Lambda(lambda x:x[:, :, 4:])(y_true)
    true_boxes = Lambda(lambda x:x[:, :, :4])(y_true)

    pred_classes = Lambda(lambda x:x[:, :, 4:])(y_pred)
    pred_boxes = Lambda(lambda x:x[:, :, :4])(y_pred)

    # Create masks for the positive and negative ground truth classes.
    negatives = Lambda(lambda x:x[:, :, -1])(true_classes)  # Tensor of shape (batch_size, n_boxes)

    positives = tf.math.less(negatives, 0.9)
    positives = tf.cast(positives, tf.float32)
    positives = tf.expand_dims(positives, axis=2)

    positive_mask = tf.concat([positives, positives, positives, positives], axis=-1)
    # positive_mask
    pred_boxes_mask = pred_boxes * positive_mask

    box_loss = smooth_L1_loss(true_boxes, pred_boxes_mask)
    class_loss = log_loss(true_classes, pred_classes)

    n_positive = tf.to_float(tf.count_nonzero(positives))

    total_loss = (box_loss + class_loss)/tf.maximum(n_positive, 1.0)

    return total_loss * batch_size


def train():

    yolo_model = get_model()
    # yolo_model.load_weights('models/trained_model_32x32_20kp_batdataset.hdf5')

    adam = k.optimizers.Adam(lr=0.001, decay=0.000005)

    yolo_model.compile(adam, loss=ssd_loss)

    # Model saving and checkpoint callbacks
    yolo_model.save(os.path.join(model_out_path, 'empty_model_7x7.hdf5'))
    filepath = os.path.join(model_out_path, "weights.best_7x7.hdf5")
    early_stopping = k.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5)
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    tensorboard = k.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard, early_stopping]

    # pose_aug_generator - Augments and reads images.  Obtains num_keypoints+rgb channels
    history = yolo_model.fit_generator(ds,
                             steps_per_epoch=ds.n_images // batch_size,
                             epochs=200,validation_data=val_ds, validation_steps=100,
                             callbacks=callbacks_list)

    yolo_model.save(os.path.join(model_out_path, 'trained_model_7x7.hdf5'))



def test():
    yolo_model = get_model()
    yolo_model.load_weights(os.path.join(model_out_path, 'trained_model_7x7.hdf5'))

    for g in ds:
        break

    out = yolo_model.predict(g[0])

    ds.get_predict2Boxes(g[0], out)


if __name__ == '__main__':
    # train()
    test()