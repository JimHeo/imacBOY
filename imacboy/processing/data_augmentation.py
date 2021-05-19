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
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing

#-----------------------------------------------------#
#               Data Augmentation class               #
#-----------------------------------------------------#
# Class to perform diverse data augmentation techniques
class Data_Augmentation:
    # Configurations for the data augmentation techniques
    config_p_per_sample = 0.15                      # Probability a data augmentation technique
                                                    # will be performed on the sample
    config_mirror_axes = "horizontal_and_vertical"
    config_contrast_range = (0.5, 1.5)
    config_gamma_range = (0.7, 1.5)
    config_rotations_range = 0.15
    config_scaling_range = (-0.15, 0.15)

    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, cycles=1, scaling=True, rotations=True, mirror=True, contrast=True):
        # Parse parameters
        self.cycles = cycles
        self.scaling = scaling
        self.rotations = rotations
        self.mirror = mirror
        self.contrast = contrast

    #---------------------------------------------#
    #            Run data augmentation            #
    #---------------------------------------------#
    def run(self, img_data, target_data):
        data_aug = Sequential()
        # Add mirror augmentation
        if self.mirror:
            data_aug.add(preprocessing.RandomFlip(self.config_mirror_axes))
        # Add contrast augmentation
        if self.contrast:
            data_aug.add(preprocessing.RandomContrast(self.config_gamma_range))
        # Add rotation augmentation
        if self.rotations:
            data_aug.add(preprocessing.RandomRotation(self.config_rotations_range, fill_mode='constant'))
        # Add scaling augmentation
        if self.scaling:
            data_aug.add(preprocessing.RandomZoom(self.config_scaling_range, self.config_scaling_range))
        
        # Perform the data augmentation x times (x = cycles)
        aug_img_data = None
        for _ in range(self.cycles):
            # Access augmentated data from the batchgenerators data structure
            if aug_img_data is None:
                aug_img_data = img_data
            # Concatenate the new data augmentated data with the cached data
            else:
                aug_img_data = np.concatenate((data_aug(img_data), aug_img_data), axis=0)
        # Return augmentated image data
        return aug_img_data, target_data
