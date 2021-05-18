import os
from data_loading.interfaces.image_io import Image_interface
from data_loading.data_io import Data_IO
from processing.data_augmentation import Data_Augmentation
from processing.subfunctions.normalization import Normalization
from processing.subfunctions.resize import Resize
from processing.preprocessor import Preprocessor
from neural_network.architecture.xception import Architecture
from neural_network.model import Neural_Network
from neural_network.metrics import binary_f1_focal, f1_score, binary_crossentropy, binary_focal_loss, precision
from evaluation.split_validation import split_validation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize the image I/O interface and configure the pascal voc images as RGB and 20 classes
interface = Image_interface(classes=20, img_type="rgb", img_format="jpg", dataset="pascal_voc", imgset='trainval')

# Specify the dataset directory
data_path = "/Users/jim/TIL/VOCdevkit/VOC2012/"
# Create the Data I/O object 
data_io = Data_IO(interface, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2)

# Create a pixel value normalization Subfunction through Z-Score 
sf_normalize = Normalization(mode='z-score')
# Create a resize Subfunction to image size 299x299
sf_resize = Resize(new_shape=(299, 299))

# Assemble Subfunction classes into a list
# Be aware that the Subfunctions will be exectued according to the list order!
subfunctions = [sf_resize, sf_normalize]

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, batch_size=4, subfunctions=subfunctions, data_aug=data_aug)

# Create the Neural Network model
unet_densePlain = Architecture(activation="softmax")
model = Neural_Network(preprocessor=pp, architecture=unet_densePlain, loss=binary_f1_focal, metrics=[f1_score, binary_crossentropy, binary_focal_loss(gamma=2., alpha=.25), precision], learninig_rate=0.0003)

# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='val_f1_score', factor=0.1, patience=20, verbose=1, mode='max', min_delta=0.0001, cooldown=1, min_lr=0.00000003)
cb_es = EarlyStopping(monitor='val_f1_score', min_delta=0, patience=150, verbose=1, mode='max')

split_validation(sample_list, model, percentage=0.2, epochs=10, evaluation_path="evaluation", draw_figures=True, run_detailed_evaluation=True, callbacks=[cb_lr, cb_es])