# this module is taken from tensorflow example repository to use functions and algorithms in the project

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code to run a pose estimation with a TFLite MoveNet model."""

import os
from typing import Dict, List

import cv2
from data import BodyPart
from data import Person
from data import person_from_keypoints_with_scores
import numpy as np

# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter
# pylint: enable=g-import-not-at-top


class Movenet(object):
  """A wrapper class for a Movenet TFLite pose estimation model."""

  # Configure how confidence the model should be on the detected keypoints to
  # proceed with using smart cropping logic.
  _MIN_CROP_KEYPOINT_SCORE = 0.2
  _TORSO_EXPANSION_RATIO = 1.9
  _BODY_EXPANSION_RATIO = 1.2

  def __init__(self, model_name: str) -> None:
    """Initialize a MoveNet pose estimation model.

    Args:
      model_name: Name of the TFLite MoveNet model.
    """

    # Append TFLITE extension to model_name if there's no extension
    _, ext = os.path.splitext(model_name)
    if not ext:
      model_name += '.tflite'

    # Initialize model
    interpreter = Interpreter(model_path=model_name, num_threads=4)
    interpreter.allocate_tensors()

    self._input_index = interpreter.get_input_details()[0]['index']
    self._output_index = interpreter.get_output_details()[0]['index']

    self._input_height = interpreter.get_input_details()[0]['shape'][1]
    self._input_width = interpreter.get_input_details()[0]['shape'][2]

    self._interpreter = interpreter
    self._crop_region = None

  def init_crop_region(self, image_height: int,
                       image_width: int) -> Dict[(str, float)]:
    if image_width > image_height:
      x_min = 0.0
      box_width = 1.0
      # Pad the vertical dimension to become a square image.
      y_min = (image_height / 2 - image_width / 2) / image_height
      box_height = image_width / image_height
    else:
      y_min = 0.0
      box_height = 1.0
      # Pad the horizontal dimension to become a square image.
      x_min = (image_width / 2 - image_height / 2) / image_width
      box_width = image_height / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }

  def _torso_visible(self, keypoints: np.ndarray) -> bool:
    left_hip_score = keypoints[BodyPart.LEFT_HIP.value, 2]
    right_hip_score = keypoints[BodyPart.RIGHT_HIP.value, 2]
    left_shoulder_score = keypoints[BodyPart.LEFT_SHOULDER.value, 2]
    right_shoulder_score = keypoints[BodyPart.RIGHT_SHOULDER.value, 2]

    left_hip_visible = left_hip_score > Movenet._MIN_CROP_KEYPOINT_SCORE
    right_hip_visible = right_hip_score > Movenet._MIN_CROP_KEYPOINT_SCORE
    left_shoulder_visible = left_shoulder_score > Movenet._MIN_CROP_KEYPOINT_SCORE
    right_shoulder_visible = right_shoulder_score > Movenet._MIN_CROP_KEYPOINT_SCORE

    return ((left_hip_visible or right_hip_visible) and
            (left_shoulder_visible or right_shoulder_visible))

  def _determine_torso_and_body_range(self, keypoints: np.ndarray,
                                      target_keypoints: Dict[(str, float)],
                                      center_y: float,
                                      center_x: float) -> List[float]:
    torso_joints = [
        BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER, BodyPart.LEFT_HIP,
        BodyPart.RIGHT_HIP
    ]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
      dist_y = abs(center_y - target_keypoints[joint][0])
      dist_x = abs(center_x - target_keypoints[joint][1])
      if dist_y > max_torso_yrange:
        max_torso_yrange = dist_y
      if dist_x > max_torso_xrange:
        max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for idx in range(len(BodyPart)):
      if keypoints[BodyPart(idx).value, 2] < Movenet._MIN_CROP_KEYPOINT_SCORE:
        continue
      dist_y = abs(center_y - target_keypoints[joint][0])
      dist_x = abs(center_x - target_keypoints[joint][1])
      if dist_y > max_body_yrange:
        max_body_yrange = dist_y

      if dist_x > max_body_xrange:
        max_body_xrange = dist_x

    return [
        max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange
    ]

  def _determine_crop_region(self, keypoints: np.ndarray, image_height: int,
                             image_width: int) -> Dict[(str, float)]:
    # Convert keypoint index to human-readable names.
    target_keypoints = {}
    for idx in range(len(BodyPart)):
      target_keypoints[BodyPart(idx)] = [
          keypoints[idx, 0] * image_height, keypoints[idx, 1] * image_width
      ]

    # Calculate crop region if the torso is visible.
    if self._torso_visible(keypoints):
      center_y = (target_keypoints[BodyPart.LEFT_HIP][0] +
                  target_keypoints[BodyPart.RIGHT_HIP][0]) / 2
      center_x = (target_keypoints[BodyPart.LEFT_HIP][1] +
                  target_keypoints[BodyPart.RIGHT_HIP][1]) / 2

      (max_torso_yrange, max_torso_xrange, max_body_yrange,
       max_body_xrange) = self._determine_torso_and_body_range(
           keypoints, target_keypoints, center_y, center_x)

      crop_length_half = np.amax([
          max_torso_xrange * Movenet._TORSO_EXPANSION_RATIO,
          max_torso_yrange * Movenet._TORSO_EXPANSION_RATIO,
          max_body_yrange * Movenet._BODY_EXPANSION_RATIO,
          max_body_xrange * Movenet._BODY_EXPANSION_RATIO
      ])

      # Adjust crop length so that it is still within the image border
      distances_to_border = np.array(
          [center_x, image_width - center_x, center_y, image_height - center_y])
      crop_length_half = np.amin(
          [crop_length_half, np.amax(distances_to_border)])

      # If the body is large enough, there's no need to apply cropping logic.
      if crop_length_half > max(image_width, image_height) / 2:
        return self.init_crop_region(image_height, image_width)
      # Calculate the crop region that nicely covers the full body.
      else:
        crop_length = crop_length_half * 2
      crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
      return {
          'y_min':
              crop_corner[0] / image_height,
          'x_min':
              crop_corner[1] / image_width,
          'y_max': (crop_corner[0] + crop_length) / image_height,
          'x_max': (crop_corner[1] + crop_length) / image_width,
          'height': (crop_corner[0] + crop_length) / image_height -
                    crop_corner[0] / image_height,
          'width': (crop_corner[1] + crop_length) / image_width -
                   crop_corner[1] / image_width
      }
    # Return the initial crop regsion if the torso isn't visible.
    else:
      return self.init_crop_region(image_height, image_width)

  def _crop_and_resize(
      self, image: np.ndarray, crop_region: Dict[(str, float)],
      crop_size: (int, int)) -> np.ndarray:
    """Crops and resize the image to prepare for the model input."""
    y_min, x_min, y_max, x_max = [
        crop_region['y_min'], crop_region['x_min'], crop_region['y_max'],
        crop_region['x_max']
    ]

    crop_top = int(0 if y_min < 0 else y_min * image.shape[0])
    crop_bottom = int(image.shape[0] if y_max >= 1 else y_max * image.shape[0])
    crop_left = int(0 if x_min < 0 else x_min * image.shape[1])
    crop_right = int(image.shape[1] if x_max >= 1 else x_max * image.shape[1])

    padding_top = int(0 - y_min * image.shape[0] if y_min < 0 else 0)
    padding_bottom = int((y_max - 1) * image.shape[0] if y_max >= 1 else 0)
    padding_left = int(0 - x_min * image.shape[1] if x_min < 0 else 0)
    padding_right = int((x_max - 1) * image.shape[1] if x_max >= 1 else 0)

    # Crop and resize image
    output_image = image[crop_top:crop_bottom, crop_left:crop_right]
    output_image = cv2.copyMakeBorder(output_image, padding_top, padding_bottom,
                                      padding_left, padding_right,
                                      cv2.BORDER_CONSTANT)
    output_image = cv2.resize(output_image, (crop_size[0], crop_size[1]))

    return output_image

  def _run_detector(
      self, image: np.ndarray, crop_region: Dict[(str, float)],
      crop_size: (int, int)) -> np.ndarray:
    input_image = self._crop_and_resize(image, crop_region, crop_size=crop_size)
    input_image = input_image.astype(dtype=np.uint8)

    self._interpreter.set_tensor(self._input_index,
                                 np.expand_dims(input_image, axis=0))
    self._interpreter.invoke()

    keypoints_with_scores = self._interpreter.get_tensor(self._output_index)
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # Update the coordinates.
    for idx in range(len(BodyPart)):
      keypoints_with_scores[idx, 0] = crop_region[
          'y_min'] + crop_region['height'] * keypoints_with_scores[idx, 0]
      keypoints_with_scores[idx, 1] = crop_region[
          'x_min'] + crop_region['width'] * keypoints_with_scores[idx, 1]

    return keypoints_with_scores

  def detect(self,
             input_image: np.ndarray,
             reset_crop_region: bool = False) -> Person:
    image_height, image_width, _ = input_image.shape
    if (self._crop_region is None) or reset_crop_region:
      # Set crop region for the first frame.
      self._crop_region = self.init_crop_region(image_height, image_width)

    # Detect pose using the crop region inferred from the detection result in
    # the previous frame
    keypoint_with_scores = self._run_detector(
        input_image,
        self._crop_region,
        crop_size=(self._input_height, self._input_width))
    # Calculate the crop region for the next frame
    self._crop_region = self._determine_crop_region(keypoint_with_scores,
                                                    image_height, image_width)

    # Convert the keypoints with scores to a Person data type

    return person_from_keypoints_with_scores(keypoint_with_scores, image_height,
                                             image_width)
