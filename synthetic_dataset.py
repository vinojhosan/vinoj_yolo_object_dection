import numpy as np
import cv2 as cv
from numpy.random import randint as randint_orig

from keras.utils import Sequence

def randint(a, b):
    if a<b:
        return randint_orig(a, b)
    elif a > b:
        return randint_orig(b, a)
    else:
        return a


class ShapeDataset(object):

    def __init__(self):
        self.min_value = 50

    def random_color(self):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        return [r, g, b]

    def get_restricted_rect(self, rect, img_shape):

        [x1, y1, x2, y2] = rect
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 >= img_shape[1]: x2 = img_shape[1] - 1
        if y2 >= img_shape[0]: y2 = img_shape[0] - 1

        return [x1, y1, x2, y2]

    def get_random_point(self, img):
        x = randint(0, img.shape[1] - self.min_value)
        y = randint(0, img.shape[0] - self.min_value)
        return [x, y]

    def get_small_side(self, img):
        return int(min(img.shape[0]/2, img.shape[1]/2))

    def draw_rectangle(self, img):

        [x1, y1] = self.get_random_point(img)
        x2 = randint(x1+self.min_value-1, img.shape[1]-1)
        y2 = randint(y1+self.min_value-1, img.shape[0] - 1)

        cv.rectangle(img, (x1, y1), (x2, y2), self.random_color(), -1)

        return img, [x1, y1, x2, y2]

    def draw_circle(self, img):
        xc = randint(0, img.shape[1] - self.min_value)
        yc = randint(0, img.shape[0] - self.min_value)

        smallest_side = int(self.get_small_side(img)/2)
        radius = randint(self.min_value, smallest_side)

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
            height = randint(self.min_value, pt[1])
        except:
            height = randint(pt[1], self.min_value)

        base_by_2 = int(randint(self.min_value, pt[0])/2)

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

        count = randint(3, count+1)
        for cnt in range(count):

            shape_int = randint(0, 3)

            if shape_int == 0:
                img, rect = self.draw_rectangle(img)
            elif shape_int == 1:
                img, rect = self.draw_circle(img)
            else:
                img, rect = self.draw_triangle(img)

            shape_rect.append([shape_int, rect])

        return img, shape_rect


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

class SynthenticGenerator(Sequence):

    def __init__(self, height, width, n_images=200, batch_size=8):
        ANCHORS = []
        self.batch_size = batch_size
        self.n_images = n_images
        self.height = height
        self.width = width

        self.anchors = [BoundBox(0, 0, ANCHORS[2 * i], ANCHORS[2 * i + 1])
                        for i in range(int(len(ANCHORS) // 2))]

        self.dataset = ShapeDataset()

    def __len__(self):
        return self.n_images // self.batch_size

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        for k in range(self.batch_size):
            img, shape_rect = self.dataset.generate_image((self.height, self.width), 10)


        return img, shape_rect


    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.image_list)



def test():
    dataset = ShapeDataset()
    img = dataset.create_image([600, 800])

    img, rect = dataset.draw_rectangle(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_circle(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_rectangle(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_triangle(img)
    img = dataset.draw_boundingbox(img, rect)

    img, rect = dataset.draw_triangle(img)
    img = dataset.draw_boundingbox(img, rect)

    cv.imwrite('shape.png', img)


if __name__ == '__main__':
    dataset = ShapeDataset()
    img, shape_rect = dataset.generate_image([300, 300], 10)
    cv.imwrite('generated_image.png', img)
    print(shape_rect)



