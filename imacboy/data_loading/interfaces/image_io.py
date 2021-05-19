#==============================================================================#
#  Original Work Author:   Dominik Müller                                      #
#  Copyright:              2020 IT-Infrastructure for                          #
#                          Translational Medical Research,                     #
#                          University of Augsburg                              #
#                                                                              #
#  Derivative Work Author: Jim Heo                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
from PIL import Image
import numpy as np
import re
# Internal libraries/scripts
from imacboy.data_loading.interfaces.abstract_io import Abstract_IO

PASCAL_VOC_DATASET = {
    'num_of_classes': 20,
    'list_of_classes': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'ignore_difficult': True,
    'train': 5717,
    'trainval': 11540,
    'val': 5823,
    'test': 10991
}

CUSTOM_DATASET = {
    'num_of_classes':-1,
    'list_of_classes':[],
    'train': 0,
    'trainval': 0,
    'val': 0
}

#-----------------------------------------------------#
#                  image I/O Interface                #
#-----------------------------------------------------#
""" Data I/O Interface for JPEG, PNG and other common 2D image files.
    Images are read by calling the imread function from the Pillow module.

Methods:
    __init__                Object creation function
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image
    load_target:            Load a target

Args:
    classes (int):          Number of classes of the dataset
    img_type (string):      Type of imaging. Options: "grayscale", "rgb"
    img_format (string):    Imaging format: Popular formats: "png", "tif", "jpg"
    pattern (regex):        Pattern to filter samples
"""
class Image_interface(Abstract_IO):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, classes=20, img_type="rgb", img_format="jpg", pattern=None, dataset="pascal_voc", imgset="trainval"):
        self.classes = classes
        self.img_type = img_type
        self.img_format = img_format
        self.three_dim = False
        self.pattern = pattern
        self.list_path = ""
        self.data_directory = ""
        self.target_cache = {}
        if img_type == "grayscale" : self.channels = 1
        elif img_type == "rgb" : self.channels = 3
        if dataset == "custom" : self.dataset = CUSTOM_DATASET
        elif dataset == "pascal_voc": 
            self.dataset = PASCAL_VOC_DATASET
            self.imgset = imgset

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    def initialize(self, input_path):
        # Resolve location where imaging data set should be located
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Cache data directory
        if self.dataset == PASCAL_VOC_DATASET: self.data_directory = os.path.join(input_path, "JPEGImages/")
        else: self.data_directory = input_path
        # Identify samples
        if self.dataset == PASCAL_VOC_DATASET:
            sample_list = []
            self.list_path = os.path.join(input_path, "ImageSets/Main/")
            with open(os.path.join(self.list_path, self.imgset + ".txt"), 'r') as f:
                for line in f.readlines():
                    sample_list.append(line.rstrip('\n'))
            for class_idx, class_str in enumerate(self.dataset['list_of_classes']):
                with open(os.path.join(self.list_path, class_str + "_" + self.imgset + ".txt"), 'r') as f:
                    for line in f.readlines():
                        img_idx, value = line.split()
                        value = int(value)
                        if value == 1 or (value == 0 if self.dataset['ignore_difficult'] == False else None):
                            if not img_idx in self.target_cache:
                                self.target_cache[img_idx] = (np.array(self.dataset['list_of_classes']) == class_str).astype(np.float32)
                            else:
                                self.target_cache[img_idx][class_idx] = 1.
                                
        else:
            sample_list = os.listdir(input_path)
        # IF pattern provided: Remove every file which does not match
        if self.pattern != None and isinstance(self.pattern, str):
            for i in reversed(range(0, len(sample_list))):
                if not re.fullmatch(self.pattern, sample_list[i]):
                    del sample_list[i]
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    def load_image(self, index):
        # Make sure that the image file exists in the data set directory
        if self.dataset == PASCAL_VOC_DATASET:
            img_path = os.path.join(self.data_directory, index + "." + self.img_format)
            if not os.path.exists(img_path):
                raise ValueError(
                    "Sample could not be found \"{}\"".format(img_path)
                )
            img_raw = Image.open(img_path)
        else:
            img_path = os.path.join(self.data_directory, index)
            if not os.path.exists(img_path):
                raise ValueError(
                    "Sample could not be found \"{}\"".format(img_path)
                )
            # Load image from file
            img_raw = Image.open(os.path.join(img_path, "imaging" + "." + self.img_format))
        # Convert image to rgb or grayscale
        if self.img_type == "grayscale":
            img_pil = img_raw.convert("LA")
        elif self.img_type == "rgb":
            img_pil = img_raw.convert("RGB")
        # Convert Pillow image to numpy matrix
        img = np.array(img_pil)
        # Keep only intensity for grayscale images
        if self.img_type =="grayscale" : img = img[:,:,0]
        # Return image
        return img

    #---------------------------------------------#
    #                 load_target                 #
    #---------------------------------------------#
    def load_target(self, index):
        if self.dataset == PASCAL_VOC_DATASET:
            target_data = self.target_cache[index]
        else:
            # Make sure that the target file exists in the data set directory
            target_path = os.path.join(self.data_directory, index)
            if not os.path.exists(target_path):
                raise ValueError(
                    "Target could not be found \"{}\"".format(target_path)
                )
        
        return target_data
    
