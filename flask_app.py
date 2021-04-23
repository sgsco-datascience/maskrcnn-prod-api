


import time
import base64

import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean

import tensorflow as tf

from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

from PIL import Image, ImageDraw

from flask import Flask, render_template, request, jsonify
import requests
import io

app = Flask(__name__)

# Lets annotate to pascal format
Image.MAX_IMAGE_PIXELS = None

global graph
########################################################################
# PATHS TO SAVE FILES AND MAKE THE DIRECTORIES
MODEL_DIR= "/datadrive/barcode-mrcnn-model/"
################################################################################################################################

# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "maskrcnn_barcode_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
#    GPU_COUNT = 0
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
#     IMAGE_MIN_DIM = 256
#     IMAGE_MAX_DIM = 512
#     DETECTION_MIN_CONFIDENCE= 0.9
#     RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)
#     RPN_NMS_THRESHOLD = 0.9
#     DETECTION_NMS_THRESHOLD= 0.0

############################################################################

cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir= MODEL_DIR, config=cfg)
# load model weights
#model.load_weights(MODEL_DIR + 'mask_rcnn_barcode_scanner.h5', by_name=True) # Prod
model.load_weights('barcode_maskrcnn.h5', by_name=True) # Test Dev


graph = tf.get_default_graph()


def normalize_images(WIDTH, HEIGHT, boxes):
    rbox = []
    for box in boxes:
        xmin = box[1] / WIDTH
        ymin = box[0] / HEIGHT
        xmax = box[3] / WIDTH
        ymax = box[2] / HEIGHT
        rbox.append([xmin, ymin, xmax, ymax] )

    return rbox


def predict_BATCH(image_mrcnn_batch, thumbnail_maxsize=1200):
    print ("==================================================================================================================================")
    
    
    for i in image_mrcnn_batch:
        i.thumbnail((thumbnail_maxsize, thumbnail_maxsize))
    
    x = [np.asarray(i) for i in image_mrcnn_batch]
    
    normalized_output = []
    #for image_mrcnn_ in image_mrcnn:
    
    with graph.as_default():
        
        yhat_batch = model.detect(x)
    for idx, yhat in enumerate(yhat_batch):
        index = np.argwhere(yhat['scores'] > 0.60).shape[0]
        classes_predicted =yhat['class_ids'][:index,]
        classes_predicted =classes_predicted[classes_predicted == 1]
        scores = yhat['scores'][:index,][classes_predicted == 1]
        bbox_predicted = yhat['rois'][:index,][classes_predicted == 1]
        masks_predicted = yhat['masks'][:,:,:index,][:,:,classes_predicted == 1]
        
        WIDTH = image_mrcnn_batch[idx].size[0]
        HEIGHT = image_mrcnn_batch[idx].size[1]
        normalized_output.append(normalize_images(WIDTH, HEIGHT, bbox_predicted))
    return normalized_output     


### NEEDED ###
@app.route('/mask_rcnn/',methods=['GET','POST'])
def scanner():
    
    #try:
    #print(type(request.data['image']))
    img = Image.open(io.BytesIO(request.data)) #PIL Image input to barcode scanner
    image_mrcnn = [img] #passing one image
    normalized_output = predict_BATCH(image_mrcnn, thumbnail_maxsize=1200)

    response = {}
    
    response['normalized_output'] = normalized_output
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9090)
    
