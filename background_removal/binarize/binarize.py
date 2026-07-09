#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os
import time
import argparse
import cv2
import warnings
import numpy as np
from tensorflow.keras import backend as K
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from . import util, utilDataGenerator, utilModelREDNet

util.init()
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

#if K.backend() == 'tensorflow':
#    import tensorflow as tf    # Memory control with Tensorflow
#    config = tf.compat.v1.ConfigProto()
#    config.gpu_options.allow_growth=True
#    sess = tf.compat.v1.Session(config=config)


# ----------------------------------------------------------------------------
def parse_menu():
    params = {}
    params['modelpath'] = 'MODELS/model_weights_all_None_256x256_s96_aug_m205_f64_k5_s2_se3_e200_b32_esp.h5'
    params['window'] = 256
    params['nb_filters'] = 64
    params['kernel'] = 5
    params['dropout'] = 0.0
    params['stride'] = 2
    params['every'] = 1
    params['threshold'] = 0.5
    params['step'] = params['window']

    return params


# ----------------------------------------------------------------------------
def build_SAE_network(config):
    nb_layers = 5
    autoencoder, encoder, decoder = utilModelREDNet.build_REDNet(nb_layers,
                                            config['window'], config['nb_filters'],
                                            config['kernel'], config['dropout'],
                                            config['stride'], config['every'])

    autoencoder.compile(optimizer='adam', loss=util.micro_fm, metrics=['mse'])

    pkg_models = os.listdir( os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELS') )
    if config['modelpath'].replace('MODELS/', '') in pkg_models:
        config['modelpath'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['modelpath'])
    autoencoder.load_weights(config['modelpath'])

    return autoencoder


# ----------------------------------------------------------------------------
def load_and_prepare_input_image(config, image):
    img = image
    assert img is not None, 'Empty file'
    img = np.asarray(img)

    original_rows = img.shape[0]
    origina_cols = img.shape[1]
    if img.shape[0] < config['window'] or img.shape[1] < config['window']:  # Scale approach
        new_rows = config['window'] if img.shape[0] < config['window'] else img.shape[0]
        new_cols = config['window'] if img.shape[1] < config['window'] else img.shape[1]
        img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

    img = np.asarray(img).astype('float32')
    img = 255. - img

    return img, original_rows, origina_cols


# ----------------------------------------------------------------------------
def run_binarize(img_in, img_out):
    args = parse_menu()
    args['imgpath'] = img_in
    args['outFilename'] = img_out

    autoencoder = build_SAE_network(args)

    input_image = cv2.imread(args['imgpath'], cv2.IMREAD_UNCHANGED)
    # Make a copy of original image to apply background removal mask at the end
    original = input_image.copy()
    # If image has alpha channel, fill transparent background with average color of image
    if len(input_image.shape) == 3 and input_image.shape[2] == 4:
        avg_color_per_row = np.average(input_image[input_image[:,:,3]>0], axis=0)
        input_image[input_image[:,:,3]==0] = avg_color_per_row
    # If image is not grayscale, convert to grayscale
    if len(input_image.shape) > 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    img, rows, cols = load_and_prepare_input_image(args, input_image)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)

    finalImg = img.copy()
    # Padding edge
    img = np.pad(img, 5, mode='edge')
    margin_size = 5

    start_time = time.time()
    # Set stepSize = step - margin_size*2
    for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=args['step']-margin_size*2, windowSize=(args['window'], args['window'])):
            if window.shape[0] != args['window'] or window.shape[1] != args['window']:
                continue

            roi = img[y:(y + args['window']), x:(x + args['window'])].copy()
            roi = roi.reshape(1, args['window'], args['window'], 1)
            roi = roi.astype('float32')

            prediction = autoencoder.predict(roi)
            prediction = (prediction > args['threshold'])
            # Keep only parts within the margin of the prediction in the final image
            finalImg[y:(y + args['window']-margin_size*2), x:(x + args['window']-margin_size*2)] = prediction[0].reshape(args['window'], args['window'])[margin_size:args['window'] - margin_size, margin_size:args['window'] - margin_size]

    print( 'Time: {:.3f} seconds'.format( time.time() - start_time ) )

    finalImg = 1. - finalImg
    finalImg *= 255.
    finalImg = finalImg.astype('uint8')

    if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
        finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

    if args['outFilename'] != None :
        # Find threshold
        threshold = np.average(finalImg[finalImg < 128]) + 10
        # Change threshold to mask
        final_mask = finalImg < threshold
        # Apply mask to original image to remove background
        output = original*final_mask[:,:,np.newaxis]
        cv2.imwrite(args['outFilename'], output)

