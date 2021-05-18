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
import numpy as np
from tensorflow.keras.utils import to_categorical
import threading
# Internal libraries/scripts
from processing.data_augmentation import Data_Augmentation
from processing.batch_creation import create_batches
from data_loading.interfaces.image_io import PASCAL_VOC_DATASET

#-----------------------------------------------------#
#                 Preprocessor class                  #
#-----------------------------------------------------#
# Class to handle all preprocessing functionalities
class Preprocessor:
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating a Preprocessor object.
    This class provides functionality for handling all preprocessing methods. This includes diverse
    optional processing subfunctions like resampling, clipping, normalization or custom subfcuntions.
    This class processes the data into batches which are ready to be used for training, prediction and validation.

    The user is only required to create an instance of the Preprocessor class with the desired specifications
    and Data IO instance (optional also Data Augmentation instance).

    Args:
        data_io (Data_IO):                      Data IO class instance which handles all I/O operations according to the user
                                                defined interface.
        batch_size (integer):                   Number of samples inside a single batch.
        subfunctions (list of Subfunctions):    List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
        data_aug (Data_Augmentation):           Data Augmentation class instance which performs diverse data augmentation techniques.
                                                If no Data Augmentation is provided, an instance with default settings will be created.
                                                Use data_aug=None, if you want no data augmentation at all.
    """
    def __init__(self, data_io, batch_size, subfunctions=[], data_aug=None):
        # Parse Data Augmentation
        if isinstance(data_aug, Data_Augmentation):
            self.data_augmentation = data_aug
        else:
            self.data_augmentation = None
        # Parse parameter
        self.data_io = data_io
        self.batch_size = batch_size
        self.subfunctions = subfunctions

    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    img_queue = []                          # Intern queue of already processed and data augmentated images.
                                            # The function create_batches will use this queue to create batches
    thread_lock = threading.Lock()          # Create a threading lock for multiprocessing

    #---------------------------------------------#
    #               Prepare Batches               #
    #---------------------------------------------#
    # Preprocess data and prepare the batches for a given list of indices
    def run(self, indices_list, training=True, validation=False):
        # Initialize storage type
        all_batches = []                    # List of batches from all samples (can sum up large amount of memory with wrong usage)
        # Iterate over all samples
        for index in indices_list:
            # Load sample and process provided subfunctions on image data
            sample = self.data_io.sample_loader(index, load_target=training)
            for sf in self.subfunctions:
                sf.preprocessing(sample)
            # Load sample from file with already processed subfunctions
            # Transform digit classes into categorical
            if training and not self.data_io.interface.dataset == PASCAL_VOC_DATASET:
                sample.target_data = to_categorical(sample.target_data, num_classes=sample.classes)
            # Decide if data augmentation should be performed
            if training and not validation and self.data_augmentation is not None:
                data_aug = True
            else:
                data_aug = False
            # Run image analysis
            ready_data = self.analysis(sample, training, data_aug)
            # Identify if current index is the last one
            if index == indices_list[-1]: last_index = True
            else : last_index = False
            # Identify if incomplete_batches are allowed for batch creation
            if training : incomplete_batches = False
            else : incomplete_batches = True
            # Create threading lock to avoid parallel access
            with self.thread_lock:
                # Put the preprocessed data at the image queue end
                self.img_queue.extend(ready_data)
                # Create batches by gathering images from the img_queue
                batches = create_batches(self.img_queue, self.batch_size, incomplete_batches, last_index)
            # Backup batches to memory
            all_batches.extend(batches)
        # Return prepared batches
        return all_batches

    def analysis(self, sample, training, data_aug):
        # Access image and target data
        img = sample.img_data
        if training : target = sample.target_data
        # Expand image dimension to simulate a batch with one image
        img_data = np.expand_dims(img, axis=0)
        if training : target_data = np.expand_dims(target, axis=0)
        # Run data augmentation
        if data_aug:
            img_data, target_data = self.data_augmentation.run(img_data, target_data)
        # Create tuple of preprocessed data
        if training : ready_data = list(zip(img_data, target_data))
        else : ready_data = list(zip(img_data))
        # Return preprocessed data tuple
        return ready_data
