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
from processing.subfunctions.abstract_subfunction import Abstract_Subfunction
from utils.augment_resize import augment_resize

#-----------------------------------------------------#
#              Subfunction class: Resize              #
#-----------------------------------------------------#
""" A Resize Subfunction class which resizes an images according to a desired shape.

Methods:
    __init__                Object creation function
    preprocessing:          Resize imaging data to the desired shape
"""
class Resize(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, new_shape=(299, 299)):
        self.new_shape = new_shape

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample):
        # Access data
        img_data = sample.img_data
        # Transform data from channel-last to channel-first structure
        img_data = np.moveaxis(img_data, -1, 0)
        # Resize imaging data
        img_data = augment_resize(img_data, self.new_shape, order=3)
        # Transform data from channel-first back to channel-last structure
        img_data = np.moveaxis(img_data, 0, -1)
        # Save resized imaging data to sample
        sample.img_data = img_data
