from keras.utils import Sequence
from config import IMAGE_H, IMAGE_W, GRID_H, GRID_W, BATCH_SIZE, BOX, ANCHORS, TRUE_BOX_BUFFER, LABELS
import cv2 as cv
import numpy as np
import numpy.random as random
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class BatchGenerator(Sequence):

    def __init__(self, file_path, img_path, shuffle=True):

        self.file_2_data_structure(img_path, file_path)
        self.n_images = len(self.image_list)

        self.anchors = [BoundBox(0, 0, ANCHORS[2 * i], ANCHORS[2 * i + 1])
                        for i in range(int(len(ANCHORS) // 2))]

        self.grid_ratio_h = float(IMAGE_H) / GRID_H
        self.grid_ratio_w = float(IMAGE_W) / GRID_W

        self.shuffle = shuffle

        self.seq = iaa.Sequential([
            iaa.Resize({"height": IMAGE_H, "width": "keep-aspect-ratio"}),
            iaa.CropToFixedSize(height=IMAGE_H, width=IMAGE_W),
            iaa.Sometimes(0.5, iaa.Multiply((0.9, 1.2))),
            iaa.Sometimes(0.5, iaa.Affine(scale=(0.75, 1.25))),
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),
        ])

        # ### augmentors by https://github.com/aleju/imgaug
        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        #
        # # Define our sequence of augmentation steps that will be applied to every image
        # # All augmenters with per_channel=0.5 will sample one value _per image_
        # # in 50% of all cases. In all other cases they will sample new values
        # # _per channel_.
        # self.aug_pipe = iaa.Sequential(
        #     [
        #         # apply the following augmenters to most images
        #         # iaa.Fliplr(0.5), # horizontally flip 50% of all images
        #         # iaa.Flipud(0.2), # vertically flip 20% of all images
        #         # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        #         sometimes(iaa.Affine(
        #             # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        #             # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        #             # rotate=(-5, 5), # rotate by -45 to +45 degrees
        #             # shear=(-5, 5), # shear by -16 to +16 degrees
        #             # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        #             # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        #             # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        #         )),
        #         # execute 0 to 5 of the following (less important) augmenters per image
        #         # don't execute all of them, as that would often be way too strong
        #         iaa.SomeOf((0, 5),
        #                    [
        #                        # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
        #                        iaa.OneOf([
        #                            iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
        #                            iaa.AverageBlur(k=(2, 7)),
        #                            # blur image using local means with kernel sizes between 2 and 7
        #                            iaa.MedianBlur(k=(3, 11)),
        #                            # blur image using local medians with kernel sizes between 2 and 7
        #                        ]),
        #                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        #                        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
        #                        # search either for all edges or for directed edges
        #                        # sometimes(iaa.OneOf([
        #                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
        #                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
        #                        # ])),
        #                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        #                        # add gaussian noise to images
        #                        iaa.OneOf([
        #                            iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
        #                            # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        #                        ]),
        #                        # iaa.Invert(0.05, per_channel=True), # invert color channels
        #                        iaa.Add((-10, 10), per_channel=0.5),
        #                        # change brightness of images (by -10 to 10 of original value)
        #                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        #                        # change brightness of images (50-150% of original value)
        #                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
        #                        # iaa.Grayscale(alpha=(0.0, 1.0)),
        #                        # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
        #                        # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
        #                    ],
        #                    random_order=True
        #                    )
        #     ],
        #     random_order=True
        # )
        #
        # if shuffle: np.random.shuffle(self.images)

    def file_2_data_structure(self, image_path, csv_path):

        df = pd.read_csv(csv_path)

        image_list = df['image_name']

        self.image_list = image_list.unique()
        self.anno_data = {}

        for img in self.image_list:
            img_info = []
            img_df = df[df['image_name'] == img]

            for tr in img_df.iterrows():

                class_name = tr[1]['class_name']
                rect = [tr[1]['xmin'], tr[1]['ymin'], tr[1]['xmax'], tr[1]['ymax']]
                img_info.append({'class_name':class_name, 'rect':rect})

            self.anno_data[img] = {'image_path': image_path + img, 'data': img_info}

    def __len__(self):
        return int(np.ceil(float(len(self.image_list)) / BATCH_SIZE))

    def num_classes(self):
        return len(LABELS)

    def size(self):
        return self.n_images

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv.imread(self.images[i]['filename'])

    def norm(self, img_batch):
        return img_batch.astype(np.float32) / 255.0 - 0.5

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    def image_augmentation(self, image, bb_list):

        bb_ia_list = [BoundingBox(r[0], r[1], r[2], r[3]) for r in bb_list]

        bbs = BoundingBoxesOnImage(bb_ia_list, shape=image.shape)
        seq_det = self.seq.to_deterministic()
        image_aug, bbs_aug = seq_det(image=image, bounding_boxes=bbs)

        out_bb = []
        for bb in bbs_aug.bounding_boxes:
            out_bb.append([bb.x1, bb.y1, bb.x2, bb.y2])

        if False:
            for i in range(len(bbs.bounding_boxes)):
                before = bbs.bounding_boxes[i]
                after = bbs_aug.bounding_boxes[i]
                print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                    i,
                    before.x1, before.y1, before.x2, before.y2,
                    after.x1, after.y1, after.x2, after.y2)
                      )

            # image with BBs before/after augmentation (shown below)
            image_before = bbs.draw_on_image(image, size=2)
            image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
            cv.imwrite(str(before.x1) + 'before.png', image_before)
            cv.imwrite(str(before.x1) + 'after.png', image_after)

        return image_aug, out_bb

    def __getitem__(self, idx):
        l_bound = idx * BATCH_SIZE
        r_bound = (idx + 1) * BATCH_SIZE

        if r_bound >= self.n_images-1:
            r_bound = self.n_images-1
            l_bound = r_bound - BATCH_SIZE

        instance_count = 0

        x_batch = np.zeros((BATCH_SIZE, IMAGE_H, IMAGE_W, 3))  # input images
        b_batch = np.zeros((BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4))  # GT boxes
        y_batch = np.zeros((BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + len(LABELS)))  # desired network output

        for t in range(l_bound,r_bound):
            # augment input image and fix object's position and size
            anno_data = self.anno_data[self.image_list[t]]
            img_path = anno_data['image_path']
            img_info = anno_data['data']
            img = cv.imread(img_path)

            rect_list = []
            class_list = []

            for obj in img_info:
                rect_list.append(obj['rect'])
                class_list.append(obj['class_name'])

            img_aug, rect_aug_list = self.image_augmentation(img, rect_list)
            true_box_index = 0
            for i, rect in enumerate(rect_list):

                # rect = obj['rect']
                class_name = class_list[i]

                center_x = .5 * (rect[0] + rect[2])
                center_x /= self.grid_ratio_w
                center_y = .5 * (rect[1] + rect[3])
                center_y /= self.grid_ratio_h

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_index = LABELS.index(class_name)

                    center_w = (rect[2] - rect[0]) / self.grid_ratio_w  # unit: grid cell
                    center_h = (rect[3] - rect[1]) / self.grid_ratio_w  # unit: grid cell

                    box = [center_x, center_y, center_w, center_h]
                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou = -1

                    shifted_box = BoundBox(0, 0, center_w, center_h)

                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou = self.bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = i
                            max_iou = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_index] = 1

                    # assign the true box to b_batch
                    b_batch[instance_count, 0, 0, 0, true_box_index] = box

                    true_box_index += 1
                    true_box_index = true_box_index % TRUE_BOX_BUFFER


            # assign input image to x_batch
            x_batch[instance_count] = self.norm(img_aug)

            # increase instance counter in current batch
            instance_count += 1

            # print(' new batch created', idx)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.image_list)


# for b in batch_generator:
#     # print(b)
#     image = b[0][0][0, :, :, :]
#
#     break
