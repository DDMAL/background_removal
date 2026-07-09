#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os
import time
import argparse
import cv2
import warnings
import numpy as np
# from tensorflow.keras import backend as K
from tensorflow.keras import backend as K

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from . import util, utilDataGenerator, utilModelREDNet

util.init()
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

# TF2: enable GPU memory growth via the modern API (the old tf.compat.v1.Session was
# never wired via set_session, so it was a no-op under eager execution).
if K.backend() == 'tensorflow':
    import tensorflow as tf
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)


# ----------------------------------------------------------------------------
def parse_menu():
    parser = argparse.ArgumentParser(description='A selectional auto-encoder approach for document image binarization')

    parser.add_argument('-imgpath',    required=True,        help='Path to the image to process')
    parser.add_argument('-modelpath',   default='MODELS/model_weights_all_None_256x256_s96_aug_m205_f64_k5_s2_se3_e200_b32_esp.h5',
                                                    help='Path to the model to load')
    parser.add_argument('-w',        default=256,    dest='window',              type=int,   help='Window size')
    parser.add_argument('-s',        default=-1,     dest='step',                type=int,   help='Step size. -1 to use window size')
    parser.add_argument('-f',        default=64,     dest='nb_filters',          type=int,   help='Nb. filters')
    parser.add_argument('-k',        default=5,      dest='kernel',              type=int,   help='Kernel size')
    parser.add_argument('-drop',     default=0,      dest='dropout',             type=float, help='Dropout value')
    parser.add_argument('-stride',   default=2,                                  type=int,   help='RED-Net stride')
    parser.add_argument('-every',    default=1,                                  type=int,   help='RED-Net shortcuts every x layers')
    parser.add_argument('-th',       default=0.5,    dest='threshold',           type=float, help='Threshold volue')
    parser.add_argument('-save',     default='out.png',   dest='outFilename',     help='Output image filename')

    args = parser.parse_args()

    if args.step == -1:
        args.step = args.window

    return args


# ----------------------------------------------------------------------------
def build_SAE_network(config):
    nb_layers = 5
    autoencoder, encoder, decoder = utilModelREDNet.build_REDNet(nb_layers,
                                            config.window, config.nb_filters,
                                            config.kernel, config.dropout,
                                            config.stride, config.every)

    autoencoder.compile(optimizer='adam', loss=util.micro_fm, metrics=['mse'])

    pkg_models = os.listdir( os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELS') )
    if config.modelpath.replace('MODELS/', '') in pkg_models:
        config.modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.modelpath)

    autoencoder.load_weights(config.modelpath)

    return autoencoder


# ----------------------------------------------------------------------------
def load_and_prepare_input_image(config):
    img = cv2.imread(config.imgpath, cv2.IMREAD_UNCHANGED)
    # Make a copy of original image to apply background removal mask at the end
    original = img.copy()
    original = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
    assert img is not None, 'Empty file'
    # If image has alpha channel, fill transparent background with average color of image
    if len(img.shape) == 3 and img.shape[2] == 4:
        avg_color_per_row = np.average(img[img[:,:,3]>0], axis=0)
        img[img[:,:,3]==0] = avg_color_per_row
    # If image is not grayscale, convert to grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)

    original_rows = img.shape[0]
    origina_cols = img.shape[1]
    if img.shape[0] < config.window or img.shape[1] < config.window:  # Scale approach
        new_rows = config.window if img.shape[0] < config.window else img.shape[0]
        new_cols = config.window if img.shape[1] < config.window else img.shape[1]
        img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

    img = np.asarray(img).astype('float32')
    img = 255. - img

    return img, original_rows, origina_cols, original


# ----------------------------------------------------------------------------
def main():
    args = parse_menu()

    autoencoder = build_SAE_network(args)

    img, rows, cols, original = load_and_prepare_input_image(args)

    finalImg = img.copy()
    # Padding edge of 5 pixels
    img = np.pad(img, 5, mode='edge')
    margin_size = 5

    start_time = time.time()
    # Set stepSize = step - margin_size*2
    for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=args.step-margin_size*2, windowSize=(args.window, args.window)):
            if window.shape[0] != args.window or window.shape[1] != args.window:
                continue

            roi = img[y:(y + args.window), x:(x + args.window)].copy()
            roi = roi.reshape(1, args.window, args.window, 1)
            roi = roi.astype('float32')

            prediction = autoencoder.predict(roi)
            prediction = (prediction > args.threshold)
            # Keep only parts within the margin of the prediction in the final image
            finalImg[y:(y + args.window-margin_size*2), x:(x + args.window-margin_size*2)] = prediction[0].reshape(args.window, args.window)[margin_size:args.window - margin_size, margin_size:args.window - margin_size]

    print( 'Time: {:.3f} seconds'.format( time.time() - start_time ) )

    finalImg = 1. - finalImg
    finalImg *= 255.
    finalImg = finalImg.astype('uint8')

    if finalImg.shape[0] != rows or finalImg.shape[1] != cols:
        finalImg = cv2.resize(finalImg, (cols, rows), interpolation = cv2.INTER_CUBIC)

    if args.outFilename != None :
        # Find threshold
        threshold = np.average(finalImg[finalImg < 128]) + 10
        # Change threshold to mask
        final_mask = finalImg < threshold
        # Apply mask to original image to remove background
        output = original*final_mask[:,:,np.newaxis]
        cv2.imwrite(args.outFilename, output)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

