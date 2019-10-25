import os

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from synthetic.synthetic_dataset import ShapeDataset


class SimpleGenerator(Sequence):
    def __init__(self, image_size, grid, batch_size, n_images=500):

        self.batch_size = batch_size
        self.n_images = n_images

        self.height = image_size[0]
        self.width = image_size[1]
        self.grid_h = grid[0]
        self.grid_w = grid[1]
        self.grid_ratio_w = self.width / self.grid_w
        self.grid_ratio_h = self.height / self.grid_h

        self.dataset = ShapeDataset()
        self.n_class = self.dataset.n_class


    def __getitem__(self, idx):
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        GRID_W = self.grid_w
        GRID_H = self.grid_h

        if r_bound >= self.n_images - 1:
            r_bound = self.n_images - 1
            l_bound = r_bound - self.batch_size

        instance_count = 0

        x_batch = np.zeros((self.batch_size, self.height, self.width, 3), np.float32)  # input images
        y_batch = np.zeros((self.batch_size, GRID_H, GRID_W, self.n_class*3), np.float32)  # desired network output


        for t in range(l_bound, r_bound):
            # augment input image and fix object's position and size

            # img_aug, rect_aug_list = self.image_augmentation(img, rect_list)
            img, shape_rect = self.dataset.generate_image((self.height, self.width), 4)
            true_box_index = 0
            for i, rect_itr in enumerate(shape_rect):

                rect = rect_itr[1]
                class_id = rect_itr[0]

                center_x = .5 * (rect[0] + rect[2])
                center_y = .5 * (rect[1] + rect[3])

                grid_x = int(center_x // self.grid_ratio_w)
                grid_y = int(center_y // self.grid_ratio_h)

                if grid_x < GRID_W and grid_y < GRID_H:

                    center_x -= grid_x * self.grid_ratio_w # Offset
                    center_y -= grid_y * self.grid_ratio_h # Offset
                    center_w = (rect[2] - rect[0]) / self.grid_ratio_w  # unit: grid cell
                    center_h = (rect[3] - rect[1]) / self.grid_ratio_h  # unit: grid cell

                    box = [center_x, center_y, center_w, center_h]

                    y_batch[instance_count, grid_y, grid_x, self.n_class * 0 + class_id] = 1
                    y_batch[instance_count, grid_y, grid_x, self.n_class * 1 + class_id] = center_x
                    y_batch[instance_count, grid_y, grid_x, self.n_class * 2 + class_id] = center_y


            # assign input image to x_batch
            x_batch[instance_count] = self.norm(img)

            # increase instance counter in current batch
            instance_count += 1

        return x_batch, y_batch

    def __len__(self):
        return self.n_images // self.batch_size

    def get_predict2Boxes(self, x_batch, y_batch):
        out_path = '/mnt/Projects/Lighthouse4_BigData/3_WorkTopics/306_SmartQualityCheck/Development/test_models/out/'
        conf = y_batch[:, :, :, 0:self.n_class]
        conf = conf > 0.5
        print('total box:',  np.sum(conf))
        predicted_objects = np.nonzero(conf)

        predicted_points = []
        for box_itr in range(len(predicted_objects[0])):
            b = predicted_objects[0][box_itr]
            r = predicted_objects[1][box_itr]
            c = predicted_objects[2][box_itr]
            ch = predicted_objects[3][box_itr]

            x_offset = y_batch[b, r, c, self.n_class * 1 + ch]
            y_offset = y_batch[b, r, c, self.n_class * 2 + ch]
            class_id = ch

            centre_x = c * self.grid_ratio_w + x_offset
            centre_y = r * self.grid_ratio_h + y_offset

            # print(x_offset, y_offset, centre_x, centre_y)
            predicted_points.append([centre_x, centre_y])

            cv.circle(x_batch[b], (int(centre_x), int(centre_y)), 3, (1, 0, 0), -1)

        x_batch_reverted = self.inverse_norm(x_batch)
        for b in range(x_batch_reverted.shape[0]):
            cv.imwrite(os.path.join(out_path, str(b).zfill(3)+'.png'), x_batch_reverted[b])
            print('image')


    def norm(self, img_batch):
        return img_batch.astype(np.float32) / 255.0 - 0.5

    def inverse_norm(self, img_batch):
        return ((img_batch + 0.5) * 255).astype(np.uint8)



if __name__ == "__main__":
    gen = SimpleGenerator((300, 300), (10, 10), 1)

    for g in gen:
        gen.get_predict2Boxes(g[0], g[1])
        break