# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from PIL import Image
import pickle

# /////////////// Corruption Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from skimage import transform, feature
from io import BytesIO

# from wand.image import Image as WandImage
# from wand.api import library as wandlibrary
# import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
from PIL import ImageDraw as draw
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from pkg_resources import resource_filename


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
"""
wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,
)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
"""

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


def line_from_points(c0, r0, c1, r1):
    if c1 == c0:
        return np.zeros((28, 28))

    # Decay function defined as log(1 - d/2) + 1
    cc, rr = np.meshgrid(np.linspace(0, 27, 28), np.linspace(0, 27, 28), sparse=True)

    m = (r1 - r0) / (c1 - c0)
    f = lambda c: m * (c - c0) + r0
    dist = np.clip(np.abs(rr - f(cc)), 0, 2.3 - 1e-10)
    corruption = np.log(1 - dist / 2.3) + 1
    corruption = np.clip(corruption, 0, 1)

    l = np.int(np.floor(c0))
    r = np.int(np.ceil(c1))

    corruption[:, :l] = 0
    corruption[:, r:] = 0

    return np.clip(corruption, 0, 1)


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruption Transforms ///////////////


class Corruptions(object):
    def __init__(self, noise, severity=1):
        self.corruption = getattr(self, noise)
        self.severity = severity

    def __call__(self, x):
        return self.corruption(x, severity=self.severity)

    def gaussian_noise(self, im, severity=1):
        self.noise = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        im = np.array(im) / 255.0
        im = np.clip(im + np.random.normal(size=im.shape, scale=self.noise), 0, 1) * 255
        return im.astype(np.float32)

    def shot_noise(self, im, severity=1):
        self.noise = [60, 25, 12, 5, 3][severity - 1]
        im = np.array(im) / 255.0
        im = np.clip(np.random.poisson(im * self.noise) / float(self.noise), 0, 1) * 255
        return im.astype(np.float32)

    def impulse_noise(self, im, severity):
        self.noise = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        im = sk.util.random_noise(np.array(im) / 255.0, mode="s&p", amount=self.noise)
        im = np.clip(im, 0, 1) * 255
        return im.astype(np.float32)

    def speckle_noise(self, im, severity=1):
        self.noise = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
        im = np.array(im) / 255.0
        im = (
            np.clip(im + im * np.random.normal(size=im.shape, scale=self.noise), 0, 1)
            * 255
        )
        return im.astype(np.float32)

    def pessimal_noise(self, im, severity=None):
        c = 10.63
        with open("pessimal_noise_matrix", "rb") as f:
            pessimal_noise_matrix = pickle.load(f)
        noise = np.random.normal(size=196) @ pessimal_noise_matrix
        scaled_noise = noise / np.linalg.norm(noise) * c / 4
        self.tiled_noise = np.tile(scaled_noise.reshape(14, 14), (2, 2))
        im = np.array(im) / 255.0
        im = np.clip(im + self.tiled_noise, 0, 1) * 255
        return im.astype(np.float32)

    def gaussian_blur(self, im, severity=1):
        self.noise = [1, 2, 3, 4, 6][severity - 1]
        im = gaussian(np.array(im) / 255.0, sigma=self.noise, multichannel=True)
        im = np.clip(im, 0, 1) * 255
        return im.astype(np.float32)

    def glass_blur(self, im, severity=1):
        self.noise = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
            severity - 1
        ]
        im = np.uint8(
            gaussian(np.array(im) / 255.0, sigma=self.noise[0], multichannel=True) * 255
        )

        # locally shuffle pixels
        for i in range(self.noise[2]):
            for h in range(28 - self.noise[1], self.noise[1], -1):
                for w in range(28 - self.noise[1], self.noise[1], -1):
                    if np.random.choice([True, False], 1)[0]:
                        dx, dy = np.random.randint(
                            -self.noise[1], self.noise[1], size=(2,)
                        )
                        h_prime, w_prime = h + dy, w + dx
                        # swap
                        im[h, w], im[h_prime, w_prime] = im[h_prime, w_prime], im[h, w]

        im = (
            np.clip(gaussian(im / 255.0, sigma=self.noise[0], multichannel=True), 0, 1)
            * 255
        )
        return im.astype(np.float32)

    def defocus_blur(self, im, severity=1):
        self.noise = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        im = np.array(im) / 255.0
        kernel = disk(radius=self.noise[0], alias_blur=self.noise[1])
        im = cv2.filter2D(im, -1, kernel)
        im = np.clip(im, 0, 1) * 255
        return im.astype(np.float32)

    def motion_blur(self, im, severity=1):
        self.noise = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        output = BytesIO()
        im.save(output, format="PNG")
        im = MotionImage(blob=output.getvalue())
        im.motion_blur(
            radius=self.noise[0], sigma=self.noise[1], angle=np.random.uniform(-45, 45)
        )
        im = cv2.imdecode(np.frombuffer(im.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        return np.clip(np.array(im), 0, 255).astype(np.float32)

    def zoom_blur(self, im, severity=1):
        self.noise = [
            np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.02),
            np.arange(1, 1.26, 0.02),
            np.arange(1, 1.31, 0.03),
        ][severity - 1]

        im = (np.array(im) / 255.0).astype(np.float32)
        out = np.zeros_like(im)
        for zoom_factor in self.noise:
            out += clipped_zoom(im, zoom_factor)

        im = (im + out) / (len(self.noise) + 1)
        return np.clip(im, 0, 1) * 255

    def fog(self, im, severity=1):
        self.noise = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][
            severity - 1
        ]
        im = np.array(im) / 255.0
        max_val = im.max()
        im = im + self.noise[0] * plasma_fractal(wibbledecay=self.noise[1])[:28, :28]
        im = np.clip(im * max_val / (max_val + self.noise[0]), 0, 1) * 255
        return im.astype(np.float32)

    def frost(self, im, severity=1):
        self.noise = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][
            severity - 1
        ]
        idx = np.random.randint(5)
        self.filename = [
            resource_filename(__name__, "frost/frost1.png"),
            resource_filename(__name__, "frost/frost2.png"),
            resource_filename(__name__, "frost/frost3.png"),
            resource_filename(__name__, "frost/frost4.jpg"),
            resource_filename(__name__, "frost/frost5.jpg"),
            resource_filename(__name__, "frost/frost6.jpg"),
        ][idx]
        self.frost = cv2.imread(self.filename, 0)

        # randomly crop and convert to rgb
        x_start, y_start = (
            np.random.randint(0, self.frost.shape[0] - 28),
            np.random.randint(0, self.frost.shape[1] - 28),
        )
        frost = self.frost[x_start : x_start + 28, y_start : y_start + 28]

        im = np.clip(self.noise[0] * np.array(im) + self.noise[1] * frost, 0, 255)
        return im.astype(np.float32)

    def snow(self, im, severity=1):
        self.noise = [
            (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
        ][severity - 1]

        im = np.array(im, dtype=np.float32) / 255.0
        snow_layer = np.random.normal(
            size=im.shape, loc=self.noise[0], scale=self.noise[1]
        )
        snow_layer = clipped_zoom(snow_layer, self.noise[2])
        snow_layer[snow_layer < self.noise[3]] = 0
        snow_layer = PILImage.fromarray(
            (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L"
        )
        output = BytesIO()
        snow_layer.save(output, format="PNG")
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(
            radius=self.noise[4],
            sigma=self.noise[5],
            angle=np.random.uniform(-135, -45),
        )

        snow_layer = (
            cv2.imdecode(
                np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            / 255.0
        )

        im = self.noise[6] * im + (1 - self.noise[6]) * np.maximum(im, im * 1.5 + 0.5)
        im = np.clip(im + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        return im.astype(np.float32)

    def spatter(self, im, severity=1):
        self.noise = [
            (0.65, 0.3, 4, 0.69, 0.6, 0),
            (0.65, 0.3, 3, 0.68, 0.6, 0),
            (0.65, 0.3, 2, 0.68, 0.5, 0),
            (0.65, 0.3, 1, 0.65, 1.5, 1),
            (0.67, 0.4, 1, 0.65, 1.5, 1),
        ][severity - 1]

        im = np.array(im, dtype=np.float32) / 255.0

        liquid_layer = np.random.normal(
            size=im.shape, loc=self.noise[0], scale=self.noise[1]
        )

        liquid_layer = gaussian(liquid_layer, sigma=self.noise[2])
        liquid_layer[liquid_layer < self.noise[3]] = 0

        m = np.where(liquid_layer > self.noise[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=self.noise[4])
        m[m < 0.8] = 0

        # mud spatter
        color = 63 / 255.0 * np.ones_like(im) * m
        im *= 1 - m

        return np.clip(im + color, 0, 1) * 255

    def contrast(self, im, severity=3):
        self.noise = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

        im = np.array(im) / 255.0
        means = np.mean(im, axis=(0, 1), keepdims=True)
        im = np.clip((im - means) * self.noise + means, 0, 1) * 255
        return im.astype(np.float32)

    def brightness(self, im, severity=1):
        self.noise = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

        im = np.array(im) / 255.0
        im = sk.color.gray2rgb(im)
        im = sk.color.rgb2hsv(im)
        im[:, :, 2] = np.clip(im[:, :, 2] + self.noise, 0, 1)
        im = sk.color.hsv2rgb(im)
        im = sk.color.rgb2gray(im)

        im = np.clip(im, 0, 1) * 255
        return im.astype(np.float32)

    def saturate(self, x, severity=1):
        self.noise = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

        x = np.array(x) / 255.0
        x = sk.color.gray2rgb(x)
        x = sk.color.rgb2hsv(x)
        x = np.clip(x * self.noise[0] + self.noise[1], 0, 1)
        x = sk.color.hsv2rgb(x)
        x = sk.color.rgb2gray(x)

        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)

    def jpeg_compression(self, x, severity=1):
        self.noise = [25, 18, 15, 10, 7][severity - 1]
        output = BytesIO()
        x.save(output, "JPEG", quality=self.noise)
        x = PILImage.open(output)

        return np.array(x).astype(np.float32)

    def pixelate(self, x, severity=1):
        self.noise = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        x = x.resize((int(28 * self.noise), int(28 * self.noise)), PILImage.BOX)
        x = x.resize((28, 28), PILImage.BOX)

        return np.array(x).astype(np.float32)

    def elastic_transform(self, image, severity=1):
        self.noise = [
            (28 * 2, 28 * 0.7, 28 * 0.1),
            (28 * 2, 28 * 0.08, 28 * 0.2),
            (28 * 0.05, 28 * 0.01, 28 * 0.02),
            (28 * 0.07, 28 * 0.01, 28 * 0.02),
            (28 * 0.12, 28 * 0.01, 28 * 0.02),
        ][severity - 1]

        image = np.array(image, dtype=np.float32) / 255.0
        shape = image.shape

        # random affine
        center_square = np.float32(shape) // 2
        square_size = min(shape) // 3
        pts1 = np.float32(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ]
        )
        pts2 = pts1 + np.random.uniform(
            -self.noise[2], self.noise[2], size=pts1.shape
        ).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape, borderMode=cv2.BORDER_CONSTANT)

        dx = (
            gaussian(
                np.random.uniform(-1, 1, size=shape),
                self.noise[1],
                mode="reflect",
                truncate=3,
            )
            * self.noise[0]
        ).astype(np.float32)
        dy = (
            gaussian(
                np.random.uniform(-1, 1, size=shape),
                self.noise[1],
                mode="reflect",
                truncate=3,
            )
            * self.noise[0]
        ).astype(np.float32)

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return (
            np.clip(
                map_coordinates(image, indices, order=1, mode="constant").reshape(
                    shape
                ),
                0,
                1,
            )
            * 255
        )

    def quantize(self, x, severity=1):
        self.noise = [5, 4, 3, 2, 1][severity - 1]
        x = np.array(x).astype(np.float32)
        x *= (2 ** self.noise - 1) / 255.0
        x = x.round()
        x *= 255.0 / (2 ** self.noise - 1)

        return x

    def shear(self, x, severity=1):
        self.noise = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]

        # Randomly switch directions
        bit = np.random.choice([-1, 1], 1)[0]
        self.noise *= bit
        aff = transform.AffineTransform(shear=self.noise)

        # Calculate translation in order to keep image center (13.5, 13.5) fixed
        a1, a2 = aff.params[0, :2]
        b1, b2 = aff.params[1, :2]
        a3 = 13.5 * (1 - a1 - a2)
        b3 = 13.5 * (1 - b1 - b2)
        aff = transform.AffineTransform(shear=self.noise, translation=[a3, b3])

        x = np.array(x) / 255.0
        x = transform.warp(x, inverse_map=aff)
        x = np.clip(x, 0, 1) * 255.0
        return x.astype(np.float32)

    def rotate(self, x, severity=1):
        self.noise = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]

        # Randomly switch directions
        bit = np.random.choice([-1, 1], 1)[0]
        self.noise *= bit
        aff = transform.AffineTransform(rotation=self.noise)

        a1, a2 = aff.params[0, :2]
        b1, b2 = aff.params[1, :2]
        a3 = 13.5 * (1 - a1 - a2)
        b3 = 13.5 * (1 - b1 - b2)
        aff = transform.AffineTransform(rotation=self.noise, translation=[a3, b3])

        x = np.array(x) / 255.0
        x = transform.warp(x, inverse_map=aff)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)

    def scale(self, x, severity=1):
        self.noise = [
            (1 / 0.9, 1 / 0.9),
            (1 / 0.8, 1 / 0.8),
            (1 / 0.7, 1 / 0.7),
            (1 / 0.6, 1 / 0.6),
            (1 / 0.5, 1 / 0.5),
        ][severity - 1]

        aff = transform.AffineTransform(scale=self.noise)

        a1, a2 = aff.params[0, :2]
        b1, b2 = aff.params[1, :2]
        a3 = 13.5 * (1 - a1 - a2)
        b3 = 13.5 * (1 - b1 - b2)
        aff = transform.AffineTransform(scale=self.noise, translation=[a3, b3])

        x = np.array(x) / 255.0
        x = transform.warp(x, inverse_map=aff)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)

    def translate(self, x, severity=1):
        self.noise = [1, 2, 3, 4, 5][severity - 1]
        bit = np.random.choice([-1, 1], 2)
        dx = self.noise * bit[0]
        dy = self.noise * bit[1]
        aff = transform.AffineTransform(translation=[dx, dy])

        x = np.array(x) / 255.0
        x = transform.warp(x, inverse_map=aff)
        x = np.clip(x, 0, 1) * 255
        return x.astype(np.float32)
