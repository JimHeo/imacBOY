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

#External libraries
from tensorflow.keras.utils import Sequence
import math
import numpy as np

## Returns a batch containing one or multiple images for training/prediction
class Dataloader(Sequence):
    # Class Initialization
    def __init__(self, sample_list, preprocessor, training=False,
                 validation=False, shuffle=False):
        # Parse sample list
        if isinstance(sample_list, list) : self.sample_list = sample_list.copy()
        elif type(sample_list).__module__ == np.__name__ :
            self.sample_list = sample_list.tolist()
        else : raise ValueError("Sample list have to be a list or numpy array!")
        # Create a working environment from the handed over variables
        self.preprocessor = preprocessor
        self.training = training
        self.validation = validation
        self.shuffle = shuffle
        self.batch_queue = []
        if not training:
            self.batch_queue = preprocessor.run(sample_list, False, False)

    # Return the next batch for associated index
    def __getitem__(self, idx):
        batch = self.generate_batch(idx)
        # Return the batch containing only an image or an image and target
        if not self.training:
            return batch[0]
        else:
            return batch[0], batch[1]

    # Return the number of batches for one epoch
    def __len__(self):
        # Number of batches is preprocessed for the single sample to predict
        if not self.training:
            return len(self.batch_queue)
        # Else number of samples is dynamic -> calculate it
        else:
            if self.preprocessor.data_augmentation is not None and not \
                self.validation:
                cycles = self.preprocessor.data_augmentation.cycles
            else:
                cycles = 1
            return math.ceil((len(self.sample_list) * cycles) / self.preprocessor.batch_size)

    # At every epoch end: Shuffle batchPointer list and reset sample_list
    def on_epoch_end(self):
        if self.shuffle and self.training:
            np.random.shuffle(self.sample_list)

    # Generate a batch during runtime
    def generate_batch(self, idx):
        # output an already generated batch if there are still batches in the queue
        if self.batch_queue : return self.batch_queue.pop(0)
        # otherwise generate a new batch
        else:
            # identify number of images required for a single batch
            if self.preprocessor.data_augmentation is not None and not self.validation:
                cycles = self.preprocessor.data_augmentation.cycles
            else:
                cycles = 1
            # Create threading lock to avoid parallel access
            with self.preprocessor.thread_lock:
                sample_size = math.ceil(self.preprocessor.batch_size / cycles)
                # access samples
                samples = self.sample_list[:sample_size]
                # move samples from top to bottom in the sample queue
                del self.sample_list[:sample_size]
                self.sample_list.extend(samples)
            # create a new batch
            batches = self.preprocessor.run(samples, self.training, self.validation)
            # Create threading lock to avoid parallel access
            with self.preprocessor.thread_lock:
                # Access a newly created batch
                next_batch = batches.pop(0)
                # Add remaining batches to batch queue
                if len(batches) > 0 : self.batch_queue.extend(batches)
            # output a created batch
            return next_batch
