import numpy as np

def get_batch_mask(batch_Y, masks):
    '''
    :param batch_Y: batch label with one-hot coding [batch_size, class_num]
    :param masks: masks of all classes with shape [class_num, last_fc_layer_size]
    :return: batch_mask with size [batch_size, last_fc_layer_size]
    '''
    class_inds = np.argmax(batch_Y, axis=1)
    batch_mask = masks[class_inds]
    return batch_mask
