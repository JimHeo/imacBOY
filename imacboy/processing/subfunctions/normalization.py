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
from imacboy.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

class Normalization(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score"):
        self.mode = mode

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample):
        # Access image
        image = sample.img_data
        # Perform z-score normalization
        if self.mode == "z-score":
            # Compute mean and standard deviation
            mean = np.mean(image)
            std = np.std(image)
            # Scaling
            image_normalized = (image - mean) / std
        # Perform MinMax normalization between [0,1]
        elif self.mode == "minmax":
            # Identify minimum and maximum
            max_value = np.max(image)
            min_value = np.min(image)
            # Scaling
            image_normalized = (image - min_value) / (max_value - min_value)
        elif self.mode == "grayscale":
            # Identify minimum and maximum
            max_value = np.max(image)
            min_value = np.min(image)
            # Scaling
            image_scaled = (image - min_value) / (max_value - min_value)
            image_normalized = np.around(image_scaled * 255, decimals=0)
        else : raise NameError("Subfunction - Normalization: Unknown modus")
        # Update the sample with the normalized image
        sample.img_data = image_normalized
