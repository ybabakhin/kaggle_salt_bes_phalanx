import keras.backend as K
import tensorflow as tf
import numpy as np


def jacard_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def jacard_coef_np(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice


def jacard_coef_loss(y_true, y_pred):
    return 1 - jacard_coef(y_true, y_pred)


def jacard_coef_loss_bce(y_true, y_pred, jacard=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + jacard_coef_loss(y_true, y_pred) * jacard


def make_loss(loss_name):
    if loss_name == 'crossentropy':
        return K.binary_crossentropy
    elif loss_name == 'jacard':
        return jacard_coef_loss
    elif loss_name == 'bce_jacard':
        def loss(y, p):
            return jacard_coef_loss_bce(y, p, jacard=0.5, bce=0.5)

        return loss
    elif loss_name == 'dice':
        return dice_coef_loss
    elif loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5)

        return loss

    elif loss_name == 'lovasz':
        def loss(y, p):
            return lovasz_loss(y, p)

        return loss
    else:
        ValueError("Unknown loss")


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def Kaggle_IoU_Precision(y_true, y_pred, threshold=0.5):
    y_pred = K.squeeze(tf.to_int32(y_pred > threshold), -1)
    y_true = K.cast(y_true[..., 0], K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    truth_areas = K.sum(y_true, axis=[1, 2])
    pred_areas = K.sum(y_pred, axis=[1, 2])
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.clip(truth_areas + pred_areas - intersection, 1e-9, 128 * 128)
    check = K.map_fn(lambda x: K.equal(x, 0), truth_areas + pred_areas, dtype=tf.bool)
    p = intersection / union
    iou = K.switch(check, p + 1., p)

    prec = K.map_fn(lambda x: K.mean(K.greater(x, np.arange(0.5, 1.0, 0.05))), iou, dtype=tf.float32)
    prec_iou = K.mean(prec)
    return prec_iou


def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-6, 1 - 1e-6)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), K.ones_like(y_pred) * K.constant(alpha),
                           K.ones_like(y_pred) * K.constant(1. - alpha))
        loss = K.mean(-1. * alpha_t * (1. - p_t) ** gamma * K.log(p_t))
        return loss

    return focal_loss_fixed


# """
# Lovasz-Softmax and Jaccard hinge loss in Tensorflow
# Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
# """

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    logits = K.log(y_pred / (1. - y_pred))
    loss = lovasz_hinge(logits, y_true, per_image=True, ignore=None)
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)

        # Fixed python3
        losses.set_shape((None,))

        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

# # --------------------------- MULTICLASS LOSSES ---------------------------


# def lovasz_softmax(probas, labels, classes='all', per_image=False, ignore=None, order='BHWC'):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
#       labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#       per_image: compute the loss per image instead of per batch
#       ignore: void class labels
#       order: use BHWC or BCHW
#     """
#     if per_image:
#         def treat_image(prob_lab):
#             prob, lab = prob_lab
#             prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
#             prob, lab = flatten_probas(prob, lab, ignore, order)
#             return lovasz_softmax_flat(prob, lab, classes=classes)
#         losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
#         loss = tf.reduce_mean(losses)
#     else:
#         loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
#     return loss


# def lovasz_softmax_flat(probas, labels, classes='all'):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
#       labels: [P] Tensor, ground truth labels (between 0 and C - 1)
#       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#     """
#     C = 1
#     losses = []
#     present = []
#     class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
#     for c in class_to_sum:
#         fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
#         if classes == 'present':
#             present.append(tf.reduce_sum(fg) > 0)
#         errors = tf.abs(fg - probas[:, c])
#         errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
#         fg_sorted = tf.gather(fg, perm)
#         grad = lovasz_grad(fg_sorted)
#         losses.append(
#             tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
#                       )
#     if len(class_to_sum) == 1:  # short-circuit mean when only one class
#         return losses[0]
#     losses_tensor = tf.stack(losses)
#     if classes == 'present':
#         present = tf.stack(present)
#         losses_tensor = tf.boolean_mask(losses_tensor, present)
#     loss = tf.reduce_mean(losses_tensor)
#     return loss


# def flatten_probas(probas, labels, ignore=None, order='BHWC'):
#     """
#     Flattens predictions in the batch
#     """
#     if order == 'BCHW':
#         probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
#         order = 'BHWC'
#     if order != 'BHWC':
#         raise NotImplementedError('Order {} unknown'.format(order))
#     C = 1
#     probas = tf.reshape(probas, (-1, C))
#     labels = tf.reshape(labels, (-1,))
#     if ignore is None:
#         return probas, labels
#     valid = tf.not_equal(labels, ignore)
#     vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
#     vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
#     return vprobas, vlabels

# def keras_lovasz_softmax(labels,probas):
#     #return lovasz_softmax(probas, labels)+binary_crossentropy(labels, probas)
#     return lovasz_softmax(probas, labels)

# model.compile(loss=keras_lovasz_softmax, optimizer="adam", metrics=["accuracy",mean_iou])
# I misread the paper. Lovasz_softmax is for multi-class segmentation. For binary tasks, you need the logit and Lovasz hinge (remove the final layer, be it softmax or sigmoid)

