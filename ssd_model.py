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

model_out_path = '/mnt/Projects/Lighthouse4_BigData/3_WorkTopics/306_SmartQualityCheck/Development/test_models/'
ds = dataset.SimpleGenerator((input_h, input_w), (grid_h, grid_w), batch_size=8, n_images=1000)


def get_model():
    input_layer = k.Input((input_h, input_w, 3), name='input_image')

    resnet50 = k.applications.ResNet50(include_top=False, input_tensor=input_layer,
                                       input_shape=(input_h, input_w, 3),
                                       weights='imagenet', pooling=None) #(input_layer)
    # resnet50.summary()
    # print(resnet50.shape)
    activation_49 = resnet50.get_layer('activation_49').output

    output_channel = ds.n_class * 3
    features = k.layers.Conv2D(output_channel, (3, 3), strides=(1, 1),
                               padding='same', activation='relu')(activation_49)

    confidence = k.layers.Conv2D(ds.n_class, (1, 1), strides=(1, 1),
                                 padding='same', activation='sigmoid')(features)
    centres = k.layers.Conv2D(2 * ds.n_class, (1, 1), strides=(1, 1),
                              padding='same', activation='relu')(features)

    output = k.layers.Concatenate(axis=-1)([confidence, centres])

    ssd_model = k.models.Model(input_layer, output)

    ssd_model.summary()

    return ssd_model

def identity_metric(y_true, y_pred):
    return k.backend.mean(y_pred)

def train():

    yolo_model = get_model()
    # yolo_model.load_weights('models/trained_model_32x32_20kp_batdataset.hdf5')

    adam = k.optimizers.Adam(lr=0.001)

    yolo_model.compile(adam, loss=ssd_loss.simple_ssd_loss, metrics=[identity_metric])

    # Model saving and checkpoint callbacks
    yolo_model.save(os.path.join(model_out_path, 'empty_model_10x10.hdf5'))
    filepath = os.path.join(model_out_path, "weights.best_pose_10x10.hdf5")
    early_stopping = k.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5)
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    tensorboard = k.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard, early_stopping]

    # pose_aug_generator - Augments and reads images.  Obtains num_keypoints+rgb channels
    yolo_model.fit_generator(ds,
                             steps_per_epoch=ds.n_images // batch_size,
                             epochs=200,
                             callbacks=callbacks_list)

    yolo_model.save(os.path.join(model_out_path, 'trained_model_10x10.hdf5'))


def test():
    yolo_model = get_model()
    yolo_model.load_weights(os.path.join(model_out_path, 'weights.best_pose_10x10.hdf5'))

    for g in ds:
        break

    out = yolo_model.predict(g[0])

    ds.get_predict2Boxes(g[0], out)
    # print('predicted out shape: ', np.array(out).shape)

test()
# train()