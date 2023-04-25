import numpy as np

def compute_bbox_size(bbox):
    xmin, xmax, ymin, ymax = bbox
    return (xmax - xmin) * (ymax - ymin)

def compute_seg_size(bin_seg):
    flagged = np.where(bin_seg)
    return len(flagged[0])
