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
import random
import pickle
from imacboy.data_loading.sample import Sample

#-----------------------------------------------------#
#                    Data IO class                    #
#-----------------------------------------------------#
# Class to handle all input and output functionality
class Data_IO:
    # Class variables
    interface = None                    # Data I/O interface
    input_path = None                   # Path to input data directory
    output_path = None                  # Path to MIScnn prediction directory
    batch_path = None                   # Path to temporary batch storage directory
    indices_list = None                 # List of sample indices after data set initialization
    seed = random.randint(0,99999999)   # Random seed if running multiple instances

    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating an object of the Data IO class.
    This class provides functionality for handling all input and output processes
    of the imaging data.

    The user is only required to create an instance of the Data IO class with the desired specifications
    and IO interface for the correct format. It is possible to create a custom IO interface for handling
    special data structures or formats.

    Args:
        interface (io_interface):   A data IO interface which inherits the abstract_io class with the following methods:
                                    initialize, load_image, load_target
        input_path (string):        Path to the input data directory, in which all imaging data have to be accessible.
        output_path (string):       Path to the output data directory, in which computed predictions will be stored. This directory
                                    will be created.
    """
    def __init__(self, interface, input_path, output_path="evalutations"):
        # Parse parameter
        self.interface = interface
        self.input_path = input_path
        self.output_path = output_path
        # Initialize Data I/O interface
        self.indices_list = interface.initialize(input_path)

    #---------------------------------------------#
    #                Sample Loader                #
    #---------------------------------------------#
    # Load a sample from the data set
    def sample_loader(self, index, load_target=True, backup=False):
        # If sample is a backup -> load it from pickle
        if backup : return self.load_sample_pickle(index)
        # Load the image with the I/O interface
        image = self.interface.load_image(index)
        # Create a Sample object
        sample = Sample(index, image, self.interface.channels, self.interface.classes)
        # IF needed read the provided target for current sample
        if load_target:
            target = self.interface.load_target(index)
            sample.add_target(target)
        # Return sample object
        return sample

    #---------------------------------------------#
    #                Sample Backup                #
    #---------------------------------------------#
    # Backup samples for later access
    def backup_sample(self, sample):
        if not os.path.exists(self.batch_path) : os.mkdir(self.batch_path)
        sample_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                                   sample.index + ".pickle")
        if not os.path.exists(sample_path):
            with open(sample_path, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load a backup sample from pickle
    def load_sample_pickle(self, index):
        sample_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                                   index + ".pickle")
        with open(sample_path,'rb') as reader:
            sample = pickle.load(reader)
        return sample

    #---------------------------------------------#
    #              Obtain all samples             #
    #---------------------------------------------#
    def get_samples(self, load_seg=False, load_pred=False):
        sample_list = []
        for index in self.indices_list:
            sample = self.sample_loader(index, load_seg=load_seg,
                                        load_pred=load_pred, backup=False)
            sample_list.append(sample)
        return sample_list

    #---------------------------------------------#
    #               Variable Access               #
    #---------------------------------------------#
    def get_indiceslist(self):
        return self.indices_list.copy()

#-----------------------------------------------------#
#               Evaluation Data Backup                #
#-----------------------------------------------------#
# Backup history evaluation as TSV (Tab Separated File) on disk
def backup_history(history, evaluation_path):
    # Opening file writer
    output_path = os.path.join(evaluation_path, "history.tsv")
    with open(output_path, "w") as fw:
        # Write the header
        header = "epoch" + "\t" + "\t".join(history.keys()) + "\n"
        fw.write(header)
        # Write data rows
        zipped_data = list(zip(*history.values()))
        for i in range(0, len(history["loss"])):
            line = str(i+1) + "\t" + "\t".join(map(str, zipped_data[i])) + "\n"
            fw.write(line)

# Backup evaluation as TSV (Tab Separated File)
def backup_evaluation(data, evaluation_path, start=False):
    # Set up the evaluation directory
    if start and not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    # Define the writing type
    if start : writer_type = "w"
    else : writer_type = "a"
    # Opening file writer
    output_path = os.path.join(evaluation_path, "detailed_validation.tsv")
    with open(output_path, writer_type) as fw:
        # Join the data together to a row
        line = "\t".join(map(str, data)) + "\n"
        fw.write(line)

# Create an evaluation subdirectory and change path
def create_directories(eval_path, subeval_path=None):
    # Create evaluation directory if necessary
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    # Create evaluation subdirectory if necessary
    if subeval_path is not None:
        # Concatenate evaluation subdirectory path if present
        subdir = os.path.join(eval_path, subeval_path)
        # Set up the evaluation subdirectory
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        # Return path to evaluation subdirectory
        return subdir
    # Return path to evaluation directory
    else : return eval_path
