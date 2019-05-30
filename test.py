import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import cv2 as cv
from data_generator import BatchGenerator, BoundBox
import yolo_model

from config import IMAGE_H, IMAGE_W, TRUE_BOX_BUFFER, GRID_H, GRID_W, BOX, CLASS, BATCH_SIZE


model = yolo_model.model_creation(IMAGE_H, IMAGE_W, TRUE_BOX_BUFFER, GRID_H, GRID_W, BOX, CLASS)

model.load_weights('trained_blood.h5')

batch_generator = BatchGenerator(r'data/full_dataset.csv',
                                 r'D:\Vinoj\HandsOnCV\object_detection\FRCNN\BCCD_Dataset-master\BCCD\JPEGImages/')

def get_bounding_box(row_vector):

    x = row_vector[0]
    y = row_vector[1]
    w = row_vector[2]
    h = row_vector[3]
    x = row_vector[0]
    classes = row_vector[5:]
    x *= batch_generator.grid_ratio_w
    y *= batch_generator.grid_ratio_h
    w *= batch_generator.grid_ratio_w
    h *= batch_generator.grid_ratio_h

    bbox = BoundBox(x-w/2, y-h/2, x+w/2, y+w/2, None, classes)

    return bbox

for b in batch_generator:
    image_list = b[0]

    out = model.predict(image_list, BATCH_SIZE)

    i = 0
    box = out[i, :, :, :, 0:4]
    conf = out[i, :, :, :, 4]
    class_conf = out[i, :, :, :, 5:]

    conf_thr = conf > 0.5
    non_zero_conf = np.nonzero(conf_thr)
    print(non_zero_conf)

    for k in range(len(non_zero_conf[0])):
        r = non_zero_conf[0][k]
        c = non_zero_conf[1][k]
        anc = non_zero_conf[2][k]

        row_vector = out[i, r, c, anc, :]

        bbox = get_bounding_box(row_vector)

        label = bbox.get_label()

        color = np.zeros(3)
        if label <= 2:
            color[label] = 255

        cv.rectangle(image_list[i], (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), color, 2)

    cv.imwrite(str(i).zfill(3)+".png", (image_list[i]+0.5)*255)

    break
