import SimpleITK as sitk
import os
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import os 
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import *
import shutil

dirs="/root/data/liver/train/ct/"
test_dir="/root/data/liver/test/ct"
# os list dirs can also work for file 
# print(os.listdir(dirs))

# spaces=[]
# paths=[*subfiles(test_dir),*subfiles(dirs)]
# for path in paths:
#     ct=sitk.ReadImage(path)
#     space=ct.GetSpacing()
#     print(space)
#     spaces.append(space)
# spaces=np.stack(spaces)
# median=np.median(spaces,axis=0)
# # [0.76757812 0.76757812 1.        ]

# print (len(subfiles(dirs)))
# print (len(subfiles(test_dir)))

from scipy import ndimage
space=[0.76757812, 0.76757812 ,1.]
path1= "/root/data/liver/train/ct/volume-6.nii"


# shapes=[]
# # shape of an image in this dataset is (Z,X,Y)
# paths=[*subfiles(test_dir),*subfiles(dirs)]
# for path in paths:
#     ct = sitk.ReadImage(path ,sitk.sitkInt16)
#     ct_array = sitk.GetArrayFromImage(ct)
#     new_shape=[ct_array.shape[0]* ct.GetSpacing()[2]/space[2],ct_array.shape[1]* ct.GetSpacing()[1]/space[1],ct_array.shape[2]* ct.GetSpacing()[0]/space[0]]
#     # ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] , ct.GetSpacing()[1]/space[1], ct.GetSpacing()[0]/space[0]), order=3)
#     shapes.append(new_shape)

# median=np.median(np.stack(shapes),axis=0)
# print (median)
# [482.5        512.00000334 512.00000334]


# print (ct_array.shape)


if __name__=="__main__":
    import SimpleITK as sitk 

    img_path1="/root/data/nnUNet/nnUNet_raw/Dataset152_LiTs/imagesTr/lt_054_0000.nii.gz"
    seg_path1="/root/data/nnUNet/nnUNet_raw/Dataset152_LiTs/labelsTr/lt_054.nii.gz"
    ct=sitk.ReadImage(img_path1)
    ct_array=sitk.GetArrayFromImage(ct)
    seg=sitk.ReadImage(seg_path1)
    seg_array=sitk.GetArrayFromImage(seg)
    fig,ax=plt.subplots(2,3 ,figsize=(30,30))# 
    print (ct_array.shape,seg_array.shape)
    plt.tight_layout()
    
    ax[0,0].imshow(ct_array[ct_array.shape[0]//2,:,:].squeeze())
    ax[0,1].imshow(ct_array[:,ct_array.shape[1]//2,:].squeeze())
    ax[0,2].imshow(ct_array[:,:,ct_array.shape[2]//2].squeeze())
    ax[1,0].imshow(ct_array[seg_array.shape[0]//2,:,:].squeeze())
    ax[1,1].imshow(ct_array[:,seg_array.shape[1]//2,:].squeeze())
    ax[1,2].imshow(ct_array[:,:,seg_array.shape[2]//2].squeeze())



