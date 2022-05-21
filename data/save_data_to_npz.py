import numpy as np
import os
import cv2

def  image2npz(image_path, mask_path, save_path, image_mode = 0, mask_mode = 0):
    for name in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, name), image_mode)
        mask = cv2.imread(os.path.join(mask_path, name), mask_mode)
        mask[mask>0] = 1

        np.savez(os.path.join(save_path, name.split('.')[0]), image=image, label=mask)
        pass

if __name__=='__main__':
    image2npz('./train_data/imgs', './train_data/masks', './npz_h5_data/train_npz')



