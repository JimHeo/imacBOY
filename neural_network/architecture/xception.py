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
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#                 Architecture class                  #
#-----------------------------------------------------#
""" The Plain variant of the popular U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D model using Keras
    create_model_3D:        Creating the 3D model using Keras
"""
class Architecture(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, activation='softmax', weights='imagenet'):
        # Parse parameter
        self.activation = activation
        self.weights = weights

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2, fine_tuning=False):
        input_tensor = Input(shape=input_shape)
        self.base_model = Xception(input_tensor=input_tensor, weights=self.weights, include_top=False, pooling='avg')
        cnn_chain = self.base_model.output
        predictions = Dense(n_labels, activation=self.activation)(cnn_chain)
        
        model = Model(inputs=self.base_model.input, outputs=predictions)
        
        if fine_tuning:
            for layer in self.base_model.layers:
                layer.trainable = False
        
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2, fine_tuning=False):
        pass
    
