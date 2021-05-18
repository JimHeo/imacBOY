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
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
from tensorflow.python.keras.utils.np_utils import to_categorical
from neural_network.metrics import binary_crossentropy
from sklearn.metrics import f1_score
from neural_network.architecture.xception import Architecture
from data_loading.dataloader import Dataloader

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network and which run the whole pipeline
class Neural_Network:
    """ Initialization function for creating a Neural Network (model) object.
    This class provides functionality for handling all model methods.
    This class runs the whole pipeline and uses a Preprocessor instance to obtain batches.

    With an initialized Neural Network model instance, it is possible to run training, prediction
    and evaluations.

    Args:
        preprocessor (Preprocessor):            Preprocessor class instance which provides the Neural Network with batches.
        architecture (Architecture):            Instance of a neural network model Architecture class instance.
        loss (Metric Function):                 The metric function which is used as loss for training.
                                                Any Metric Function defined in Keras, in neural_network.metrics or any custom
                                                metric function, which follows the Keras metric guidelines, can be used.
        metrics (List of Metric Functions):     List of one or multiple Metric Functions, which will be shown during training.
                                                Any Metric Function defined in Keras, in miscnn.neural_network.metrics or any custom
                                                metric function, which follows the Keras metric guidelines, can be used.
        learning_rate (float):                  Learning rate in which weights of the neural network will be updated.
        batch_queue_size (integer):             The batch queue size is the number of previously prepared batches in the cache during runtime.
        Number of workers (integer):            Number of workers/threads which preprocess batches during runtime.
    """
    def __init__(self, preprocessor, architecture=Architecture(),
                 loss=binary_crossentropy, metrics=[f1_score],
                 learninig_rate=0.0001, batch_queue_size=10,
                 workers=1, fine_tuning=False):
        # Identify data parameters
        self.three_dim = preprocessor.data_io.interface.three_dim
        self.channels = preprocessor.data_io.interface.channels
        self.classes = preprocessor.data_io.interface.classes
        # Cache parameter
        self.preprocessor = preprocessor
        self.loss = loss
        self.metrics = metrics
        self.learninig_rate = learninig_rate
        self.batch_queue_size = batch_queue_size
        self.workers = workers
        self.output_activation = architecture.activation
        self.fine_tuning = fine_tuning
        # Build model
        self.build_model(architecture)
        # Cache starting weights
        self.initialization_weights = self.model.get_weights()


    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    shuffle_batches = True                  # Option whether batch order should be shuffled or not
    initialization_weights = None           # Neural Network model weights for weight reinitialization

    #---------------------------------------------#
    #                Model Creation               #
    #---------------------------------------------#
    def build_model(self, architecture):
        # Assemble the input shape
        input_shape = (None,)
        # Initialize model for 3D data
        if self.three_dim:
            input_shape = (None, None, None, self.channels)
            self.model = architecture.create_model_3D(input_shape=input_shape,
                                                      n_labels=self.classes, fine_tuning=self.fine_tuning)
         # Initialize model for 2D data
        else:
             input_shape = (None, None, self.channels)
             self.model = architecture.create_model_2D(input_shape=input_shape,
                                                       n_labels=self.classes, fine_tuning=self.fine_tuning)
        # Compile model
        self.model.compile(optimizer=Adam(lr=self.learninig_rate),
                           loss=self.loss, metrics=self.metrics)

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    """ Fitting function for the Neural Network model using the provided list of sample indices.

    Args:
        sample_list (list of indices):          A list of sample indicies which will be used for training
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through
                                                the complete data set.
        iterations (integer):                   Number of iterations (batches) in a single epoch.
        callbacks (list of Callback classes):   A list of Callback classes for custom evaluation
    """
    def train(self, sample_list, epochs=20, callbacks=[]):
        # Initialize Keras Dataloader for generating batches
        dataGen = Dataloader(sample_list, self.preprocessor, training=True, validation=False, shuffle=self.shuffle_batches)
        # Run training process with Keras fit
        if not self.fine_tuning:
            self.model.fit(dataGen,
                           epochs=epochs,
                           callbacks=callbacks,
                           workers=self.workers,
                           max_queue_size=self.batch_queue_size)
        else:
            pre_epochs = int(math.ceil(float(epochs * 0.2)))
            post_epochs = epochs - pre_epochs
            self.model.fit(dataGen,
                           epochs=pre_epochs,
                           callbacks=callbacks,
                           workers=self.workers,
                           max_queue_size=self.batch_queue_size)
            for layer in self.model.layers[:-30]:
                layer.trainable = False
            for layer in self.model.layers[-30:]:
                layer.trainable = True
            # Compile model
            self.model.compile(optimizer=Adam(lr=self.learninig_rate * 0.1), loss=self.loss, metrics=self.metrics)
            self.model.fit(dataGen,
                           epochs=post_epochs,
                           callbacks=callbacks,
                           workers=self.workers,
                           max_queue_size=self.batch_queue_size)

    #---------------------------------------------#
    #                 Prediction                  #
    #---------------------------------------------#
    """ Prediction function for the Neural Network model. The fitted model will predict a segmentation
        for the provided list of sample indices.

    Args:
        sample_list (list of indices):  A list of sample indicies for which a segmentation prediction will be computed
        return_output (boolean):        Parameter which decides, if computed predictions will be output as the return of this
                                        function or if the predictions will be saved with the save_prediction method defined
                                        in the provided Data I/O interface.
        activation_output (boolean):    Parameter which decides, if model output (activation function, normally softmax) will
                                        be saved/outputed (if FALSE) or if the resulting class label (argmax) should be outputed.
    """
    def predict(self, sample_list, activation_output=False):
        # Initialize result array for direct output
        results = []
        # Iterate over each sample
        for sample in sample_list:
            # Initialize Keras Dataloader for generating batches
            dataGen = Dataloader([sample], self.preprocessor, training=False, validation=False, shuffle=False)
            # Run prediction process with Keras predict
            pred_list = []
            for batch in dataGen:
                pred_batch = self.model.predict_on_batch(batch)
                pred_list.append(pred_batch)
            pred = np.concatenate(pred_list, axis=0)
            if activation_output == False:
                if self.output_activation == "softmax": pred = to_categorical(np.argmax(pred, axis=-1), num_classes=self.classes)
                elif self.output_activation == "sigmoid": pred = (pred > 0.5).astype(np.float32)
            # Backup predicted segmentation
            results.append(pred)
            
        # Output predictions results
        return results

    #---------------------------------------------#
    #                 Evaluation                  #
    #---------------------------------------------#
    """ Evaluation function for the Neural Network model using the provided lists of sample indices
        for training and validation. It is also possible to pass custom Callback classes in order to
        obtain more information.

    Args:
        training_samples (list of indices):     A list of sample indicies which will be used for training
        validation_samples (list of indices):   A list of sample indicies which will be used for validation
        epochs (integer):                       Number of epochs. A single epoch is defined as one iteration through the complete data set.
        iterations (integer):                   Number of iterations (batches) in a single epoch.
        callbacks (list of Callback classes):   A list of Callback classes for custom evaluation
    Return:
        history (Keras history object):         Gathered fitting information and evaluation results of the validation
    """
    # Evaluate the Neural Network model using the MIScnn pipeline
    def evaluate(self, training_samples, validation_samples, epochs=20, callbacks=[]):
        # Initialize a Keras Dataloader for generating Training data
        dataGen_training = Dataloader(training_samples, self.preprocessor, training=True, validation=False, shuffle=self.shuffle_batches)
        # Initialize a Keras Dataloader for generating Validation data
        dataGen_validation = Dataloader(validation_samples, self.preprocessor, training=True, validation=True, shuffle=self.shuffle_batches)
        
        self.model.compile(optimizer=Adam(lr=self.learninig_rate), loss=self.loss, metrics=self.metrics)
        # Run training & validation process with the Keras fit
        if not self.fine_tuning:
            history = self.model.fit(dataGen_training,
                                     validation_data=dataGen_validation,
                                     callbacks=callbacks,
                                     epochs=epochs,
                                     workers=self.workers,
                                     max_queue_size=self.batch_queue_size)
        else:
            pre_epochs = int(math.ceil(float(epochs * 0.2)))
            post_epochs = epochs - pre_epochs
            history = self.model.fit(dataGen_training,
                                     validation_data=dataGen_validation,
                                     callbacks=callbacks,
                                     epochs=pre_epochs,
                                     workers=self.workers,
                                     max_queue_size=self.batch_queue_size)
            for layer in self.model.layers[:-30]:
                layer.trainable = False
            for layer in self.model.layers[-30:]:
                layer.trainable = True
            # Compile model
            self.model.compile(optimizer=Adam(lr=self.learninig_rate * 0.1), loss=self.loss, metrics=self.metrics)
            history = self.model.fit(dataGen_training,
                                     validation_data=dataGen_validation,
                                     callbacks=callbacks,
                                     epochs=post_epochs,
                                     workers=self.workers,
                                     max_queue_size=self.batch_queue_size)
            
        # Return the training & validation history
        return history

    #---------------------------------------------#
    #               Model Management              #
    #---------------------------------------------#
    # Re-initialize model weights
    def reset_weights(self):
        self.model.set_weights(self.initialization_weights)

    # Dump model to file
    def dump(self, file_path):
        self.model.save(file_path)

    # Load model from file
    def load(self, file_path, custom_objects={}):
        # Create model input path
        self.model = load_model(file_path, custom_objects, compile=False)
        # Compile model
        self.model.compile(optimizer=Adam(lr=self.learninig_rate),
                           loss=self.loss, metrics=self.metrics)
