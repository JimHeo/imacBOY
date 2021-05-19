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
import numpy as np

#-----------------------------------------------------#
#                 Image Sample - class                #
#-----------------------------------------------------#
# Object containing an image and the associated target
class Sample:
    # Initialize class variable
    index = None
    img_data = None
    target_data = None
    shape = None
    channels = None
    classes = None

    # Create a Sample object
    def __init__(self, index, image, channels, classes):
        # Preprocess image data if required
        if image.shape[-1] != channels:
            image = np.reshape(image, image.shape + (channels,))
        # Cache data
        self.index = index
        self.img_data = image
        self.channels = channels
        self.classes = classes
        self.shape = self.img_data.shape

    # Add target
    def add_target(self, target):
        self.target_data = target
    
