#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import scipy.ndimage as ndi
from random import uniform, choice
import os
import cv2
import argparse
import util

util.init()

# ------------------------------------------------------------------------------
def flip(img, ftype='vertical'):
    if ftype=='vertical':
        return cv2.flip(img.copy(), 1)
    elif ftype=='horizontal':
        return cv2.flip(img.copy(), 0)
    else:
        raise('Unknown flip type')


# ------------------------------------------------------------------------------
def resize(img, scale):
    return cv2.resize(img.copy(), None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)


# ------------------------------------------------------------------------------
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
def rotate(img, rotation, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        r: Rotation degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    x = np.asarray(img.copy())
    x = x.reshape(1, x.shape[0], x.shape[1])
    #print(x.shape)

    theta = np.pi / 180 * rotation
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = __transform_matrix_offset_center(rotation_matrix, h, w)
    x = __apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    return x


# ------------------------------------------------------------------------------
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
def shear(img, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
    x = np.asarray(img.copy())
    x = x.reshape(1, x.shape[0], x.shape[1])

    shear = intensity #np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = __transform_matrix_offset_center(shear_matrix, h, w)
    x = __apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    return x


# ------------------------------------------------------------------------------
def __transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


# ------------------------------------------------------------------------------
def __apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)

    x = x.reshape(x.shape[1], x.shape[2])
    x = np.array(x, dtype=np.uint8)

    return x


# ------------------------------------------------------------------------------
def whitening(img):
    t = 5
    x = img.copy()
    x[:] = [[max(pixel - t, 0) if pixel < 127.5 else min(pixel + t, 255) for pixel in row] for row in x[:]]
    return x


# ------------------------------------------------------------------------------
def blur(img):
    return cv2.medianBlur(img.copy(), 5)


# ------------------------------------------------------------------------------
def sharpening(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img2 = cv2.filter2D(img.copy(), -1, kernel)
    return cv2.medianBlur(img2, 3)


# ------------------------------------------------------------------------------
def bilateralFilter(img):
    return cv2.bilateralFilter(img.copy(), 9, 75, 75)



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Data augmentation')
parser.add_argument('-path', required=True, help='path to dataset')
parser.add_argument('-out', required=True, help='output path')
args = parser.parse_args()

x_sufix = '_GR/'
y_sufix = '_GT/'

print('# Processing path:', args.path)

array_images = sorted( util.list_files(args.path, ext='png') )

for fname_x in array_images:
    print('· Procesing image', fname_x, '...')

    fname_y = fname_x.replace(x_sufix, y_sufix)

    img_x = cv2.imread(fname_x, False)
    img_y = cv2.imread(fname_y, False)

    arr_x = {'o': img_x}
    arr_y = {'o': img_y}

    # flips
    arr_x['v']  = flip(img_x, 'vertical')
    #arr_x['h']  = flip(img_x, 'horizontal')
    #arr_x['hv'] = flip(arr_x['h'], 'vertical')

    arr_y['v']  = flip(img_y, 'vertical')
    #arr_y['h']  = flip(img_y, 'horizontal')
    #arr_y['hv'] = flip(arr_y['h'], 'vertical')

    # scale
    for k in arr_x.keys():
        zo = uniform( *choice([(0.5, 0.9), (1.1, 1.5)]) )
        arr_x[k + '_zo'] = resize(arr_x[k], zo)
        arr_y[k + '_zo'] = resize(arr_y[k], zo)

    base_keys = arr_x.keys()  # save the flips and scale keys

    # rotate
    """for k in base_keys:
        ro = uniform( *choice([(-8, -2), (2, 8)]) )
        arr_x[k + '_ro'] = rotate(arr_x[k], ro)
        arr_y[k + '_ro'] = rotate(arr_y[k], ro)

    # shear
    for k in base_keys:
        sh = uniform( *choice([(-0.2,-0.05), (0.05,0.2)]) )
        arr_x[k + '_sh'] = shear(arr_x[k], sh)
        arr_y[k + '_sh'] = shear(arr_y[k], sh)

    # whitening, blur, sharpening
    for k in arr_x.keys():
        arr_x[k + '_w'] = whitening(      arr_x[k])
        #arr_x[k + '_b'] = blur(           arr_x[k])  # similar a f
        arr_x[k + '_s'] = sharpening(     arr_x[k])
        arr_x[k + '_f'] = bilateralFilter(arr_x[k])

        arr_y[k + '_w'] = arr_y[k]
        #arr_y[k + '_b'] = arr_y[k]
        arr_y[k + '_s'] = arr_y[k]
        arr_y[k + '_f'] = arr_y[k]"""


    # Save
    out_y = args.out.replace(x_sufix, y_sufix)

    util.mkdirp(args.out)
    util.mkdirp(out_y)

    filename, extension = os.path.splitext(os.path.basename(fname_x))
    fullname_x = os.path.join( args.out +'/' + filename )
    fullname_y = os.path.join( out_y +'/' + filename )

    for k in arr_x.keys():
        if k=='o':
            continue
        cv2.imwrite(fullname_x + '_' + k + extension, arr_x[k])
        cv2.imwrite(fullname_y + '_' + k + extension, arr_y[k])



