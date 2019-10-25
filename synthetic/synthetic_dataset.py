import numpy as np
import cv2 as cv
from numpy.random import randint

from tensorflow.keras.utils import Sequence

def random_integer(a, b):
    if a<b:
        return randint(a, b)
    elif a > b:
        return randint(b, a)
    else:
        return a


class ShapeDataset(object):

    def __init__(self):
        self.min_value = 50
        self.n_class = 3

    def random_color(self):
        r = random_integer(0, 255)
        g = random_integer(0, 255)
        b = random_integer(0, 255)
        return [r, g, b]

    def get_restricted_rect(self, rect, img_shape):

        [x1, y1, x2, y2] = rect
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 >= img_shape[1]: x2 = img_shape[1] - 1
        if y2 >= img_shape[0]: y2 = img_shape[0] - 1

        return [x1, y1, x2, y2]

    def get_random_point(self, img):
        x = random_integer(0, img.shape[1] - self.min_value)
        y = random_integer(0, img.shape[0] - self.min_value)
        return [x, y]

    def get_small_side(self, img):
        return int(min(img.shape[0]/2, img.shape[1]/2))

    def draw_rectangle(self, img):

        [x1, y1] = self.get_random_point(img)
        x2 = random_integer(x1 + self.min_value - 1, img.shape[1] - 1)
        y2 = random_integer(y1 + self.min_value - 1, img.shape[0] - 1)

        cv.rectangle(img, (x1, y1), (x2, y2), self.random_color(), -1)

        return img, [x1, y1, x2, y2]

    def draw_square(self, img):
        [x1, y1] = self.get_random_point(img)
        size = random_integer(self.min_value - 1, img.shape[1] - 1)

        cv.rectangle(img, (x1, y1), (x1+size, y1+size), self.random_color(), -1)

        return img, [x1, y1, x1+size, y1+size]

    def draw_circle(self, img):
        xc = random_integer(0, img.shape[1] - self.min_value)
        yc = random_integer(0, img.shape[0] - self.min_value)

        smallest_side = int(self.get_small_side(img)/2)
        radius = random_integer(self.min_value, smallest_side)

        cv.circle(img, (xc, yc), int(radius), self.random_color(), -1)

        x1 = xc - radius
        y1 = yc - radius
        x2 = xc + radius
        y2 = yc + radius

        rect_restricted = self.get_restricted_rect([x1, y1, x2, y2], img.shape)

        return img, rect_restricted

    def draw_triangle(self, img):

        pt = self.get_random_point(img)
        pt2 = self.get_random_point(img)
        pt3 = self.get_random_point(img)

        try:
            height = random_integer(self.min_value, pt[1])
        except:
            height = random_integer(pt[1], self.min_value)

        base_by_2 = height # int(randint(self.min_value, pt[0])/2)

        top = [pt[0], pt[1] - height]
        side_left = [pt[0] - base_by_2, pt[1]]
        side_right = [pt[0] + base_by_2, pt[1]]

        triangle_cnt = np.array([top, side_left, side_right])
        cv.drawContours(img, [triangle_cnt], 0, self.random_color(), -1)

        x1 = side_left[0]
        y1 = top[1]
        x2 = side_right[0]
        y2 = side_left[1]

        restricted_rect = self.get_restricted_rect([x1, y1, x2, y2], img.shape)

        return img, restricted_rect

    def create_image(self, size):
        img = np.ones([size[0], size[1], 3])
        [r, g, b] = self.random_color()
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        return img

    def draw_boundingbox(self, img, rect):
        [x1, y1, x2, y2] = rect
        cv.rectangle(img, (x1, y1), (x2, y2), [255, 0, 0], 1)
        return img

    def generate_image(self, size, count):

        img = self.create_image(size)
        shape_rect = []

        count = random_integer(3, count + 1)
        for cnt in range(count):

            shape_int = random_integer(0, 3)

            if shape_int == 0:
                img, rect = self.draw_square(img)
            elif shape_int == 1:
                img, rect = self.draw_circle(img)
            else:
                img, rect = self.draw_triangle(img)

            shape_rect.append([shape_int, rect])

        return img, shape_rect


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, class_index=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.class_index = class_index

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.class_index)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.class_index[self.get_label()]

        return self.score


class SynthenticGenerator(Sequence):

    def __init__(self, image_size, grid, n_images=1000, batch_size=8):
        # ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        ANCHORS = [1, 1,
                   2, 2,
                   3, 3]
        self.batch_size = batch_size
        self.n_images = n_images
        self.height = image_size[0]
        self.width = image_size[1]
        self.shuffle = True

        self.anchors = [BoundBox(0, 0, ANCHORS[2 * i], ANCHORS[2 * i + 1])
                        for i in range(int(len(ANCHORS) // 2))]

        self.dataset = ShapeDataset()

        self.n_class = self.dataset.n_class

        self.grid_h = grid[0]
        self.grid_w = grid[1]

        self.grid_ratio_w = self.width / self.grid_w
        self.grid_ratio_h = self.height / self.grid_h

    def __len__(self):
        return self.n_images // self.batch_size

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        GRID_W = self.grid_w
        GRID_H = self.grid_h

        if r_bound >= self.n_images-1:
            r_bound = self.n_images-1
            l_bound = r_bound - self.batch_size

        instance_count = 0

        x_batch = np.zeros((self.batch_size, self.height, self.width, 3), np.float32)  # input images
        y_batch = np.zeros((self.batch_size, GRID_H, GRID_W, 4 + self.n_class + 1), np.float32)  # desired network output

        # confidence_for_non_class
        y_batch[..., -1] = 1

        for t in range(l_bound,r_bound):
            # augment input image and fix object's position and size

            # img_aug, rect_aug_list = self.image_augmentation(img, rect_list)
            img, shape_rect = self.dataset.generate_image((self.height, self.width), 3)
            for i, rect_itr in enumerate(shape_rect):

                rect = rect_itr[1]
                class_name = rect_itr[0]

                center_x = .5 * (rect[0] + rect[2])
                center_y = .5 * (rect[1] + rect[3])

                grid_x = int(center_x // self.grid_ratio_w)
                grid_y = int(center_y // self.grid_ratio_h)

                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_index = class_name

                    center_x -= grid_x * self.grid_ratio_w
                    center_y -= grid_y * self.grid_ratio_h
                    center_w = (rect[2] - rect[0]) / self.grid_ratio_w  # unit: grid cell
                    center_h = (rect[3] - rect[1]) / self.grid_ratio_h  # unit: grid cell

                    box = [center_x, center_y, center_w, center_h]

                    # assign ground truth x, y, w, h and class probs to y_batch
                    y_batch[instance_count, grid_y, grid_x, 0:4] = box
                    # y_batch[instance_count, grid_y, grid_x, i, 4] = 1.
                    y_batch[instance_count, grid_y, grid_x, 4 + obj_index] = 1
                    y_batch[instance_count, grid_y, grid_x, -1] = 0

                    # print(box, rect)
                    # find the anchor that best predicts this box
                    # best_anchor = -1
                    # max_iou = -1

                    # shifted_box = BoundBox(0, 0, center_w, center_h)
                    #
                    # for i in range(len(self.anchors)):
                    #     anchor = self.anchors[i]
                    #     iou = self.bbox_iou(shifted_box, anchor)
                    #
                    #     if iou >= 0.5:
                    #         best_anchor = i
                    #
                    #         # assign ground truth x, y, w, h and class probs to y_batch
                    #         y_batch[instance_count, grid_y, grid_x, i, 0:4] = box
                    #         # y_batch[instance_count, grid_y, grid_x, i, 4] = 1.
                    #         y_batch[instance_count, grid_y, grid_x, i, 4 + obj_index] = 1
                    #         y_batch[instance_count, grid_y, grid_x, i, -1] = 0
                    #
                    # true_box_index += 1

            # assign input image to x_batch
            x_batch[instance_count] = self.norm(img)

            # increase instance counter in current batch
            # print(true_box_index)
            instance_count += 1
        box_reshaped = np.reshape(y_batch[:, :, :, 0:4], (self.batch_size, -1, 4))
        class_reshaped = np.reshape(y_batch[:, :, :, 4:], (self.batch_size, -1, self.dataset.n_class + 1))

        y_batch = np.concatenate([box_reshaped, class_reshaped], axis=-1)
        return x_batch, np.array(y_batch)

    def get_predict2Boxes(self, x_batch, y_batch):

        box_reshaped = np.reshape(y_batch[:, :, 0:4], (x_batch.shape[0], self.grid_h, self.grid_w, 4))
        class_reshaped = np.reshape(y_batch[:, :, 4:], (x_batch.shape[0], self.grid_h, self.grid_w, self.dataset.n_class + 1))

        y_batch = np.concatenate([box_reshaped, class_reshaped], axis=-1)

        conf = y_batch[:, :, :, -1]
        conf = conf < 0.5
        print('total box:',  np.sum(conf))
        predicted_objects = np.nonzero(conf)

        predicted_boxes = []
        for box_itr in range(len(predicted_objects[0])):
            b = predicted_objects[0][box_itr]
            r = predicted_objects[1][box_itr]
            c = predicted_objects[2][box_itr]
            # a = predicted_objects[3][box_itr]

            box = y_batch[b, r, c, 0:4]
            class_id = np.argmax(y_batch[b, r, c, 4:])

            centre_x = box[0] + c * self.grid_ratio_w
            centre_y = box[1] + r * self.grid_ratio_h
            center_w = box[2] * self.grid_ratio_w
            center_h = box[3] * self.grid_ratio_h

            x1 = int(centre_x - center_w / 2)
            y1 = int(centre_y - center_h / 2)
            x2 = int(centre_x + center_w / 2)
            y2 = int(centre_y + center_h / 2)
            predicted_boxes.append([x1, y1, x2, y2])

            print(box, [x1, y1, x2, y2])
            cv.rectangle(x_batch[b], (x1, y1), (x2, y2), (1, 0, 0), 3)

        x_batch_reverted = self.inverse_norm(x_batch)
        for b in range(x_batch_reverted.shape[0]):
            cv.imwrite(str(b).zfill(3)+'.png', x_batch_reverted[b])

    def norm(self, img_batch):
        return img_batch.astype(np.float32) / 255.0 - 0.5

    def inverse_norm(self, img_batch):
        return ((img_batch + 0.5) * 255).astype(np.uint8)

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




def test():
    dataset = ShapeDataset()
    img = dataset.create_image([600, 800])

    img, rect = dataset.draw_square(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_circle(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_square(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_triangle(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_triangle(img)
    img = dataset.draw_boundingbox(img, rect)

    cv.imwrite('shape.png', img)


if __name__ == '__main__':

    syn_gen = SynthenticGenerator((224, 224), (7, 7), batch_size=1)

    x_batch, y_batch = syn_gen.__getitem__(0)

    syn_gen.get_predict2Boxes(x_batch, y_batch)



