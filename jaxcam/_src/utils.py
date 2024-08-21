# Copyright 2024 The jaxcam Authors.
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

"""Camera transformation utilities.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import jaxcam
from jaxcam._src import math
import optax


def rts_to_sim3(
    rotation: jnp.ndarray, translation: jnp.ndarray, scale: float
) -> jnp.ndarray:
  """Converts a rotation, translation and scale to a homogeneous transform.

  Args:
    rotation: (3, 3) An orthonormal rotation matrix.
    translation: (3,) A 3-vector representing a translation.
    scale: A scalar factor.

  Returns:
    (4, 4) A homogeneous transformation matrix.
  """

  transform = jnp.eye(4)
  transform = transform.at[:3, :3].set(rotation * scale)
  transform = transform.at[:3, 3].set(translation)

  return transform


def sim3_to_rts(
    transform: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Converts a homogeneous transform to rotation, translation and scale.

  Args:
    transform: (4, 4) A homogeneous transformation matrix.

  Returns:
    rotation: (3, 3) An orthonormal rotation matrix.
    translation: (3,) A 3-vector representing a translation.
    scale: A scalar factor.
  """

  eps = jnp.float32(jnp.finfo(jnp.float32).tiny)
  rotation_scale = transform[..., :3, :3]
  # Assumes rotation is an orthonormal transform, thus taking norm of first row.
  scale = optax.safe_norm(rotation_scale, min_norm=eps, axis=1)[0]
  rotation = rotation_scale / scale
  translation = transform[..., :3, 3]
  return rotation, translation, scale


def relativize_cameras(
    reference: jaxcam.Camera, target: jaxcam.Camera
) -> tuple[jaxcam.Camera, jaxcam.Camera]:
  """Transforms the target camera to be in the reference camera's frame.

  Relativizes the camera such that the target cameras are in the reference
  camera's reference frame. I.e., the reference camera's camera coordinate frame
  becomes the world coordinate frame.

  Args:
    reference: The reference camera.
    target: The target camera.

  Returns:
    A tuple containing the transformed reference and target cameras. The
    reference camera will have an identity world-to-camera matrix.
  """

  transform = jaxcam.relative_transform(reference, target)
  rel_reference = jaxcam.update_world_to_camera_matrix(reference, jnp.eye(4))
  rel_target = jaxcam.update_world_to_camera_matrix(target, transform)
  return rel_reference, rel_target


def transform_camera(
    camera: jaxcam.Camera, transform: jax.Array
) -> jaxcam.Camera:
  """Transform cameras.

  Args:
    camera: The camera to be transformed.
    transform: (4, 4) Transformation matrix comprised of rotation, translation
      and scale.

  Returns:
    Transformed cameras.
  """
  rotation, translation, scale = sim3_to_rts(transform)
  transform = rts_to_sim3(rotation, translation, 1.0)
  camera_from_world = jaxcam.world_to_camera_matrix(camera)

  camera_from_transform = math.matmul(
      camera_from_world, transform[None]
  ).squeeze(0)
  camera = jaxcam.update_world_to_camera_matrix(camera, camera_from_transform)
  camera = jaxcam.scale(camera, scale)

  return camera


def transform_to_identity_rotation(
    camera: jaxcam.Camera,
) -> jaxcam.Camera:
  """Transforms a camera to have an identity rotation."""
  rotation = camera.orientation.T
  transform = rts_to_sim3(rotation, jnp.zeros(3), 1.0)
  return transform_camera(camera, transform)


def relativize_rotation(
    reference: jaxcam.Camera, target: jaxcam.Camera
) -> tuple[jaxcam.Camera, jaxcam.Camera]:
  """Transforms the target camera to be in the reference camera's frame.

  Relativizes the camera such that the target cameras are in the reference
  camera's reference frame. I.e., the reference camera's camera coordinate frame
  becomes the world coordinate frame.

  Args:
    reference: The reference camera.
    target: The target camera.

  Returns:
    A tuple containing the transformed reference and target cameras. The
    reference camera will have an identity world-to-camera matrix.
  """

  transform = reference.orientation.T
  transform = rts_to_sim3(transform, jnp.zeros(3), 1.0)
  rel_reference = transform_camera(reference, transform)
  rel_target = transform_camera(target, transform)
  return rel_reference, rel_target
