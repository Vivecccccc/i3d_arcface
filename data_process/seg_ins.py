import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

from utils.misc_utils import compute_seg_size

class SegmentBody:
    def __init__(self, ckpt='sam_vit_b_01ec64.pth', device='cuda'):
        ckpt_dir = './ckpt'
        self.sam_checkpoint = os.path.join(ckpt_dir, ckpt)
        self.model_type = {'sam_vit_h_4b8939.pth': 'vit_h', 
                           'sam_vit_l_0b3195.pth': 'vit_l',
                           'sam_vit_b_01ec64.pth': 'vit_b'}[ckpt]
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
    
    def make_segment(self, img, kpss):
        self.predictor.set_image(img)
        masks_lst = []
        for pinpoints in kpss:
            masks, scores, _ = self.predictor.predict(
                point_coords=pinpoints,
                point_labels=np.array([1] * pinpoints.shape[0]),
                multimask_output=True,
            )
            largest_mask_id = np.array(list(map(compute_seg_size, masks))).argmax()
            masks_lst.append(masks[largest_mask_id])
        return masks_lst
    
    def blackout_background(self, img, mask):
        blackout_img = np.zeros_like(img)
        if mask is not None:
            blackout_img[mask] = img[mask]
        return blackout_img

    def _show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def _show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def _show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
        