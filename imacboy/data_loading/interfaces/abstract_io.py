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
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#       Abstract Interface for the Data IO class      #
#-----------------------------------------------------#
""" An abstract base class for a Data_IO interface.

Methods:
    __init__                Object creation function
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image
    load_target:            Load a target
"""
class Abstract_IO(ABC):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    """ Functions which will be called during the I/O interface object creation.
        This function can be used to pass variables in the custom I/O interface.
        The only required passed variable is the number of channels in the images,
        the number of classes in the segmentation and the dimension of the data.

        Parameter:
            channels (integer):    Number of channels of the image (grayscale:1, RGB:3)
            classes (integer):     Number of classes in the segmentation (binary:2, multi-class:3+)
            three_dim (boolean):   Variable to express, if the data is two or three dimensional
        Return:
            None
    """
    @abstractmethod
    def __init__(self, channels=1, classes=2, three_dim=False):
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim
        pass
    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    """ Initialize and prepare the image data set, return the number of samples in the data set

        Parameter:
            input_path (string):    Path to the input data directory, in which all imaging data have to be accessible
        Return:
            indices_list [list]:    List of indices. The Data_IO class will iterate over this list and
                                    call the load_image and load_segmentation functions providing the current index.
                                    This can be used to train/predict on just a subset of the data set.
                                    e.g. indices_list = [0,1,9]
                                    -> load_image(0) | load_image(1) | load_image(9)
    """
    @abstractmethod
    def initialize(self, input_path):
        pass
    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    """ Load the image with the index i from the data set and return it as a numpy matrix.
        Be aware that MIScnn only supports a last_channel structure.
        2D: (x,y,channel) or (x,y)
        3D: (x,y,z,channel) or (x,y,z)

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            image [numpy matrix]:   A numpy matrix/array containing the image
    """
    @abstractmethod
    def load_image(self, i):
        pass
    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    """ Load the target data of the image with the index i from the data set and return it as a numpy matrix.
        Be aware that MIScnn only supports a last_channel structure.
        2D: (x,y,channel) or (x,y)
        3D: (x,y,z,channel) or (x,y,z)

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            target [numpy matrix]:     A numpy matrix/array containing the target
    """
    @abstractmethod
    def load_target(self, i):
        pass
