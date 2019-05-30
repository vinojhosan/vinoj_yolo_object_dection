import keras as k
import tensorflow as tf
import numpy as np

from config import GRID_H, GRID_W, BATCH_SIZE, BOX, CLASS, \
    ANCHORS, WARM_UP_BATCHES

from config import NO_OBJECT_SCALE, OBJECT_SCALE, COORD_SCALE, CLASS_SCALE, CLASS_WEIGHTS


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    """
    Adjust prediction
    """
    ### adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

    ### adjust w and h
    true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES),
                                                   lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                            true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                                                                ANCHORS, [1, 1, 1, BOX, 2]) * no_boxes_mask,
                                                            tf.ones_like(coord_mask)],
                                                   lambda: [true_box_xy,
                                                            true_box_wh,
                                                            coord_mask])

    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """
    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall)

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

    return loss


def my_custom_loss(y_true, y_pred):
    """
    Usually the input size of the network's output will be [None, 13, 13, 5, 9]
    [Batch size, Grid H, Grid W, Box(Anchors), confidence(1) + BBox(4) + Classes(labels:3)+ background(1)]
    :param y_true:
    :param y_pred:
    :return:
    """

    """True data label distribution"""
    true_conf = y_true[..., 0:1]
    true_xy = y_true[..., 1:3]
    true_wh = y_true[..., 3:5]
    true_classes = y_true[..., 5:5 + CLASS+1]
    true_classes_rearranged = tf.reshape(true_classes, [BATCH_SIZE * GRID_H * GRID_W * BOX, CLASS + 1])

    true_mask = tf.cast(true_conf > 0.5, tf.float32)
    true_no_obj_mask = tf.cast(true_conf < 0.5, tf.float32)
    true_mask_2_channel = tf.concat([true_mask, true_mask], axis=-1)
    true_mask_n_channel = tf.cast(true_classes_rearranged[:, CLASS] < 0.5, tf.float32)

    """Pred data label distribution"""
    pred_conf = tf.sigmoid(y_pred[..., 0:1])
    pred_xy = tf.sigmoid(y_pred[..., 1:3])
    pred_wh = tf.exp(y_pred[..., 3:5])
    pred_classes = tf.nn.softmax(y_pred[..., 5:5 + CLASS+1])
    pred_classes_rearranged = tf.reshape(pred_classes, [BATCH_SIZE * GRID_H * GRID_W * BOX, CLASS + 1])


    loss_xy = COORD_SCALE * tf.reduce_sum(tf.square(true_xy - pred_xy )* true_mask_2_channel)
    loss_wh = COORD_SCALE * tf.reduce_sum(tf.square(true_wh - pred_wh) * true_mask_2_channel)
    loss_conf = OBJECT_SCALE * tf.reduce_sum(tf.square(true_conf - pred_conf) * true_mask)
    loss_class_temp = tf.nn.softmax_cross_entropy_with_logits(labels=true_classes_rearranged,
                                                         logits=pred_classes_rearranged)
    loss_class_temp = k.backend.categorical_crossentropy(target=true_classes_rearranged,
                                                         output=pred_classes_rearranged)
    loss_class = tf.reduce_sum(loss_class_temp * true_mask_n_channel)

    no_obj_loss = NO_OBJECT_SCALE * tf.reduce_sum(tf.square(true_conf - pred_conf) * true_no_obj_mask)

    loss = loss_xy + loss_wh + loss_conf + loss_class + no_obj_loss
    
    loss = loss / (BATCH_SIZE) 

    if False:
        loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
        loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
        loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
        loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
        loss = tf.Print(loss, [no_obj_loss], message='no_obj_loss \t', summarize=1000)
        loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
        loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    return loss

if False:
    import data_generator as dg
    batch_generator = dg.BatchGenerator(r'data/full_dataset.csv',
                                 r'/home/vinoj/vinoj/keras-frcnn/JPEGImages/')

    for data in batch_generator:
        break

    y_true = data[1]

    loss = my_custom_loss(y_true, y_true)
    sess = tf.Session()
    print(sess.run(loss))
