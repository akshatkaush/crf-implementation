import os
import cv2
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from imgaug import augmenters as iaa

def crf(or_im, mask_im, n_classes=2):
    seq = iaa.Sequential([
    iaa.SaltAndPepper(0.05)
    ])

    # remove if image already in rgb channel
    or_im = cv2.cvtColor(or_im, cv2.COLOR_BGR2RGB) 

    not_mask = cv2.bitwise_not(mask_im)
    not_mask = np.expand_dims(not_mask, axis=2)
    mask_im = np.expand_dims(mask_im, axis=2)

    im_softmax = np.concatenate([not_mask, mask_im], axis=2)
    im_softmax = im_softmax / 255.0

    gauss_im = cv2.GaussianBlur(or_im, (31, 31), 0)
    bilat_im = cv2.bilateralFilter(or_im, d=10, sigmaColor=80, sigmaSpace=80)

    feat_first = im_softmax.transpose((2, 0, 1)).reshape((n_classes,-1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(or_im.shape[1], or_im.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(10, 10), srgb=(2, 2, 2), rgbim=or_im,
                        compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((or_im.shape[0], or_im.shape[1]))

    res_hot = to_categorical(res) * 255.0

    crf_mask = np.array(res*255, dtype=np.uint8)

    return crf_mask

def main():
    cat_bgr = cv2.imread('2.jpeg', 1)
    cat_mask = cv2.imread('mask.jpg', 0)
    crf_im=crf(cat_bgr, cat_mask)

    display(Image.fromarray(crf_im))
    
if __name__ == "__main__":
    main()