import os
import numpy as np
import cv2
import h5py
from PIL import Image

#img_mode、mask_mode:  1:rgb(cv2.IMREAD_COLOR)，0:gray(cv2.IMREAD_GRAYSCALE)
def save_image_to_h5py(img_path, mask_path, save_path, name='case', img_mode=0, mask_mode=0):
    path_list = [img_path, mask_path]
    img_list = []
    mask_list = []

    for dir_image in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, dir_image), img_mode)
        mask = cv2.imread(os.path.join(mask_path, dir_image), mask_mode)
        mask[mask>0] = 1

        img_list.append(img)
        mask_list.append(mask)


    img_np = np.array(img_list, dtype=np.float32)
    label_np = np.array(mask_list, dtype=np.float32)

    f = h5py.File(os.path.join(save_path, name+'.npy.h5'), 'w')
    f['image'] = img_np
    f['label'] = label_np
    f.close()

if __name__=='__main__':
   save_image_to_h5py('./test_data/imgs', './test_data/masks',
                       './npz_h5_data/test_vol_h5', name='case')

def load_h5py_to_np(path, shuffle=True):
    h5_file = h5py.File(path, 'r')
    if shuffle==True:
        permutation = np.random.permutation(len(h5_file['label']))
    else:
        permutation = np.array(range(len(h5_file['label'])))
    shuffled_image = h5_file['image'][:][permutation]
    shuffled_label = h5_file['label'][:][permutation]

    return shuffled_image, shuffled_label

# if __name__=='__main__':
#     images, labels = load_h5py_to_np('./npz_h5_data/test_vol_h5/case.npy.h5')
#     # images, labels = load_h5py_to_np(r'D:\Downloads\project_TransUNet\data\Synapse\test_vol_h5\case0008.npy.h5')
#     # images, labels = images*255, labels*255
#
#     for image, mask in zip(images, labels):
#         result = Image.new('F', (image.shape[0]*2, image.shape[1]))
#         result.paste(Image.fromarray(image),
#                      box=(0, 0, image.shape[0], image.shape[1]))
#         result.paste(Image.fromarray(mask),
#                      box=(image.shape[0], 0, mask.shape[0]*2, mask.shape[1]))
#         result.show()
#         cv2.imwrite("filename.png", np.array(result))
