# batchgenerators is inhouse package, can view by command pipshow
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import os 
import nibabel as nib
import random
import matplotlib.pyplot as plt
import SimpleITK as sitk 
import numpy as np
from scipy import ndimage
import cv2
plt.rcParams['image.cmap'] = 'gray'
def convert_lits(lits_base_dir:str,nnunet_dataset_id: int = 168):
    task_name = "LiTs"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    img_names=subfiles(join(lits_base_dir,"ct"),join=False)
    seg_names=subfiles(join(lits_base_dir,"label"),join=False)
    img_names.sort()

    BASE="lt"
    idx=0
    pairs=[]
    for name in img_names:
        basename=os.path.splitext(name)[0]
        number=basename.split("-")[1]
        seg_name=f"segmentation-{number}"
        pairs.append(tuple([basename,seg_name])) 

  
    for (img,seg) in pairs:
        print(f"copy {img} correspond to index {idx}")
        img_path= join(lits_base_dir,"ct",f"{img}.nii")
        img_save= join(imagestr,f"{BASE}_{idx:03d}_0000.nii.gz")
        seg_path=join(lits_base_dir,"label",f"{seg}.nii")
        seg_save=join(labelstr,f"{BASE}_{idx:03d}.nii.gz")

        img_ct,seg_ct=resample(img_path,seg_path)
        sitk.WriteImage(img_ct, img_save)
        sitk.WriteImage(seg_ct,seg_save)
       
        idx+=1
    generate_dataset_json(out_base, {0: "CT"}, # here is channel name ?
                          labels={
                              "background": 0,
                              "liver": 1,                          
                              "tumor": 2
                          },
                          num_training_cases=len(pairs), file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='prerelease',
                        #   overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="LiTS")

    print ("finish")
def visualize_multiple(bases,img_dir,seg_dir,num_random_sample=4,slice_pos=[1/2,2/3,3/4],png_path="/root/repo/liver-tumor-segmentation/nnUNet/nnunetv2/learn/visualize.png"):
    def nnUnetNormalize(ct_array) :   
        ct_array[ct_array<=-17]=-17
        ct_array[ct_array>=201]=201
        ct_array=(ct_array-99.4)/39.39
        return ct_array
    def min_max_normalize(ct_array):
        ct_array[ct_array<0]=0
        ct_array[ct_array>400]=400
        ct_array=ct_array/400
        return ct_array

    plt.rcParams['image.cmap'] = 'gray'
    pair_path={}
    for i, key in enumerate(bases.keys()):
        (img_base,seg_base)=bases[key]
        pair_path[str(i)]=tuple([join(img_dir,img_base),join(seg_dir,seg_base)])
    
    fig,ax=plt.subplots(len(pair_path)*len(slice_pos),3 ,figsize=(30,30))# 
    plt.tight_layout()
    for i,key in enumerate(pair_path.keys()):
        (img_path,seg_path)=pair_path[key]
        print ("here",img_path,seg_path)
        ct=sitk.ReadImage(img_path)
        ct_array=sitk.GetArrayFromImage(ct)
        seg=sitk.ReadImage(seg_path)
        seg_array=sitk.GetArrayFromImage(seg)

        for j,pos in enumerate(slice_pos):
            img_slice=ct_array[ int(ct_array.shape[0]*pos),:,:]
            seg_slice=seg_array[int(seg_array.shape[0]*pos),:,:]
            ax[ j+i*len(slice_pos),0].imshow(img_slice)
            ax[ j+i*len(slice_pos),0].set_title(f"sample {str(key)} slice pos {str(pos)}")
            ax[ j+i*len(slice_pos),1].imshow(img_slice)
            ax[ j+i*len(slice_pos),1].imshow(seg_slice,alpha=0.5)
            ax[ j+i*len(slice_pos),2].imshow(seg_slice)

            j+i*len(slice_pos)
           


        # ax[i,0].imshow(ct_array[ct_array.shape[0]//2,:,:].squeeze())
        # ax[i,0].set_title(f"sample {str(key)}")
        
        # ax[i,2].imshow(seg_array[2*seg_array.shape[0]//3,:,:].squeeze())
        # ax[i,1].imshow(ct_array[2*ct_array.shape[0]//3,:,:].squeeze())
        # ax[i,1].imshow(seg_array[2*seg_array.shape[0]//3,:,:].squeeze(),alpha=0.5)
    
    plt.show()
    plt.savefig(png_path)
    plt.clf()


def visualize(d,root="/root/data/liver/train/",save_path="/root/repo/liver-tumor-segmentation/nnUNet/nnunetv2/learn/single.png"):
    #d is dictionary
    img_root=join(root,"ct")
    seg_root=join(root,"label")

    fig,ax=plt.subplots(len(d.keys()),3,figsize=(30,30))
    for i,key in enumerate(d.keys()):
        img_path=join(img_root,d[key][0])
        seg_path=join(seg_root,d[key][1])
        ct=sitk.ReadImage(img_path,sitk.sitkInt16)
        ct_array=sitk.GetArrayFromImage(ct)
        seg=sitk.ReadImage(seg_path,sitk.sitkUInt8)
        seg_array=sitk.GetArrayFromImage(seg)
      
        ax[i,0].imshow(ct_array[ct_array.shape[0]//2,:,:].squeeze())
        ax[i,0].set_title(f"{key}")
        ax[i,1].imshow(ct_array[ct_array.shape[0]//2,:,:].squeeze())
        ax[i,1].imshow(seg_array[seg_array.shape[0]//2,:,:].squeeze(),alpha=0.5)      
        ax[i,2].imshow(seg_array[seg_array.shape[0]//2,:,:].squeeze())

    plt.show()
    plt.savefig(save_path)
    plt.clf()

def getMetaData(pairs,img_dir,seg_dir):
    d={}  
    for key in pairs.keys():
        img,seg=pairs[key][0],pairs[key][1]
        print (f"read from {join(img_dir,img)} and {join(seg_dir,seg)}")
        ct=sitk.ReadImage(join(img_dir,img))
        seg=sitk.ReadImage(join(seg_dir,seg))
        d[img]={}
        d[img]["ct"]={}
        d[img]["seg"]={}
        d[img]["ct"]["space"]=ct.GetSpacing()
        d[img]["seg"]["space"]=seg.GetSpacing()
        d[img]["ct"]["origin"]=ct.GetOrigin()
        d[img]["seg"]["origin"]=seg.GetOrigin()
        d[img]["ct"]["direction"]=ct.GetDirection()
        d[img]["seg"]["direction"]=seg.GetDirection()
    return d

def resample (ct_path, seg_path, classes=2,spatial_zoom=1/2,depth_zoom=1,HU_crop=False,HU_up=400,HU_down=0,size=75,expand_slice=20):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array =sitk.GetArrayFromImage(ct) 
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        if classes==2:
            seg_array[seg_array > 0] = 1
        # HU intensity crop
        if HU_crop:
            ct_array[ct_array >HU_up] = HU_up
            ct_array[ct_array < HU_down] = HU_down
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / depth_zoom, spatial_zoom, spatial_zoom), order=3)
        #e.g,ct_array.shape # (375, 256, 256), 375=75*5, 
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / depth_zoom, spatial_zoom, spatial_zoom), order=0)
        # Find the starting and ending slices of the liver region and expand them in both directions."
        z = np.any(seg_array, axis=(1, 2)) 
        #axis = plane xy
        # print ("z",z.shape) # (375,) 
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        # numpy.where(condition, [x, y, ]/)
        #Return elements chosen from x or y depending on condition.#
     # np where return a tuple , if not specify more,the first one is the list of index in original array
     # where condition is True, the second onece is False, here [0] is for get the liver region
     # [[0, -1]] is get the first and last xy plane 
        #"Expand in each direction by a certain number of voxels.

        if start_slice - expand_slice < 0:
            
            start_slice = 0
        else:
            start_slice -= expand_slice

        if end_slice + expand_slice >= seg_array.shape[0]:
            end_slice = seg_array.shape[0] - 1
        else:
            end_slice += expand_slice

        print("Cut out range:",str(start_slice) + '--' + str(end_slice))
         # "If the remaining number of slices is insufficient to reach the desired 
         #size, discard the data. As a result, there will be very few instances of data."
        # size is probably theminimum number of slices
        if end_slice - start_slice + 1 < size:
            return None,None,None
        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]
        print("Preprocessed shape:",ct_array.shape,seg_array.shape)
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / spatial_zoom), ct.GetSpacing()[1] * int(1 / spatial_zoom), depth_zoom))
       
        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / spatial_zoom), ct.GetSpacing()[1] * int(1 / spatial_zoom), depth_zoom))
       
        return new_ct, new_seg

if __name__=="__main__":
    
    convert_lits(lits_base_dir="/root/data/liver/train/",nnunet_dataset_id = 168)
    
    d2={
        "53":["volume-48.nii","segmentation-48.nii"],
        "54":["volume-49.nii","segmentation-49.nii"],
        "52":["volume-58.nii","segmentation-58.nii"],

    }
    d1={
        "53":["volume-48.nii","segmentation-48.nii"],
        "54":["volume-49.nii","segmentation-49.nii"],
    }
    # visualize(["lt_054_0000.nii.gz","lt_053_0000.nii.gz","lt_058_0000.nii.gz"])
    
    # visualize(d1)
    # img="/root/data/liver/train/ct/volume-48.nii"
    # seg="/root/data/liver/train/label/segmentation-48.nii"
    # resample(img,seg)
    # visualize_multiple(d2,img_dir="/root/data/liver/train/ct",seg_dir="/root/data/liver/train/label")

    # print (getMetaData(d2,img_dir="/root/data/liver/train/ct",seg_dir="/root/data/liver/train/label"))
    