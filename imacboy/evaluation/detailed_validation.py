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
from imacboy.data_loading.data_io import backup_evaluation
from sklearn.metrics import f1_score

#-----------------------------------------------------#
#                 Detailed Validation                 #
#-----------------------------------------------------#
""" Function for detailed validation of a validation sample data set. The segmentation
    of these samples will be predicted with an already fitted model and evaluated.
Args:
    model (Neural Network model):           Instance of an already fitted Neural Network model class instance.
    validation_samples (list of indices):   A list of sample indicies which will be used for validation.
    evaluation_path (string):               Path to the evaluation data directory. This directory will be created and
                                            used for storing all kinds of evaluation results during the validation processes.
                                            if the evaluations will be saved on disk in the evaluation directory.
"""

######todo: complete the function
def detailed_validation(validation_samples, model, evaluation_path):
    # Initialize detailed validation scoring file
    classes = list(map(lambda x: "f1_class-" + str(x),
                       range(model.preprocessor.data_io.interface.classes)))
    header = ["sample_id"]
    header.extend(classes)
    backup_evaluation(header, evaluation_path, start=True)
    # Iterate over each sample
    for sample_index in validation_samples:
        # Predict the sample with the provided model
        pred = model.predict([sample_index])[0].tolist()
        # Load the sample
        sample = model.preprocessor.data_io.sample_loader(sample_index, load_target=True)
        # Access image, truth and predicted segmentation data
        target = sample.target_data
        # Calculate classwise dice score
        f1_scores = compute_f1_score(target, pred)
        # Save detailed validation scores to file
        scores = [sample_index]
        scores.extend(f1_scores)
        backup_evaluation(scores, evaluation_path, start=False)
        
def compute_f1_score(target, pred):
    score = f1_score(target, pred)
        
    return score