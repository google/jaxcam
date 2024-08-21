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

"""Utilities for converting cameras to and from ray-based representations."""

from typing import Any, Optional
from flax import struct
import jax
import jax.numpy as jnp
from jaxcam._src import camera as jaxcam
from jaxcam._src import math


@struct.dataclass
class Rays:
  """Represents cameras as a bundle of rays.

  Supports rays represented as directions and origins or directions and moments
  (Plücker coordinates [1]). Whereas a line can be represented by any origin
  along the ray, it is represented by a unique (normalized) direction and
  moment.

  This class should not be instantiated directly. Use the `create()` function
  instead.

  [1]: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates

  Attributes:
    directions: (..., 3) Ray directions.
    origins: (..., 3) Ray origins.
    moments: (..., 3) Plücker moments.
    shape: (int, ...) The shape of the rays.
  """

  directions: jnp.ndarray
  origins: jnp.ndarray

  @classmethod
  def create(
      cls,
      directions: jnp.ndarray,
      *,
      origins: Optional[jnp.ndarray] = None,
      moments: Optional[jnp.ndarray] = None,
  ) -> 'Rays':
    """Creates a Rays object.

    Note that if the rays are instantiated with moments, the origins could be
    any point along the ray. In other words,
    `Rays.create(rays.directions, moments=rays.moments).origins` is not
    guaranteed to be the same as `rays.origins`.

    Args:
      directions: (..., 3) Unit normalized ray directions.
      origins: (..., 3) Ray origins.
      moments: (..., 3) Plücker moments.

    Returns:
      A Rays object.
    """
    # Ensure directions are unit normalized.
    norms = jnp.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / norms
    if origins is not None and moments is not None:
      raise ValueError('Cannot specify both origins and moments.')
    if origins is None and moments is None:
      raise ValueError('Must specify either origins or moments.')

    if moments is not None:
      if directions.shape != moments.shape:
        raise ValueError(
            'Directions and moments must have the same shape. Got '
            f'{directions.shape} and {moments.shape}'
        )
      origins = jnp.cross(directions, moments, axis=-1)
    elif origins is not None:
      if directions.shape != origins.shape:
        raise ValueError(
            'Directions and origins must have the same shape. Got '
            f'{directions.shape} and {origins.shape}'
        )
    return cls(directions=directions, origins=origins)

  @property
  def moments(self) -> jnp.ndarray:
    norm = jnp.linalg.norm(self.directions, axis=-1, keepdims=True)
    return jnp.cross(self.origins, self.directions / norm, axis=-1)

  def to_raymap_6d(self, use_plucker: bool = False) -> jnp.ndarray:
    """Returns a 6D raymap representation of the rays.

    Raymap represents rays as [moments, directions] (Plücker coordinates) or
    [origins, directions]

    Args:
      use_plucker: If True, uses Plücker coordinates.

    Returns:
      A (..., 6) raymap representation of the rays.
    """
    if use_plucker:
      return jnp.concatenate((self.moments, self.directions), axis=-1)
    else:
      return jnp.concatenate((self.origins, self.directions), axis=-1)

  @classmethod
  def from_raymap_6d(
      cls,
      raymap: jnp.ndarray,
      use_plucker: bool = False,
  ) -> 'Rays':
    """Creates a Rays object from a raymap representation."""
    moments_or_origins = raymap[..., :3]
    directions = raymap[..., 3:]
    if use_plucker:
      moments, origins = moments_or_origins, None
    else:
      moments, origins = None, moments_or_origins
    return cls.create(
        directions=directions,
        moments=moments,
        origins=origins,
    )

  @property
  def shape(self) -> tuple[int, ...]:
    return self.directions.shape


def get_rays_from_camera(
    camera: jaxcam.Camera,
    normalize: bool = True,
) -> Rays:
  """Computes rays unprojected from every pixel.

  Currently only supports a single camera.

  Args:
    camera: Camera to unproject.
    normalize (bool): If True, normalizes the directions to have unit norm.

  Returns:
    A tuple of (directions, origins) where both are (H, W, 3).
  """
  pixels = jaxcam.get_pixel_centers(*camera.image_size)
  directions = jaxcam.pixels_to_rays(camera, pixels, normalize=normalize)
  origins = jnp.tile(
      camera.position, (int(camera.image_size[1]), int(camera.image_size[0]), 1)
  )
  return Rays.create(directions=directions, origins=origins)


def get_camera_from_rays(
    rays: Rays,
    use_ransac: bool = False,
    ransac_parameters: Optional[dict[str, Any]] = None,
) -> jaxcam.Camera:
  """Recovers a Camera from a ordered grid of rays.

  Performs a least-squares fit for a pinhole camera using DLT. See Sec 4.1 of
  Hartley and Zisserman [1] for more details. Note that the returned camera may
  have parameters that are slightly different from the original camera due to
  numerical precision.

  [1]
  https://github.com/DeepRobot2020/books/blob/master/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf

  Args:
    rays: Camera rays where directions and origins are (H, W, 3).
    use_ransac: If True, uses RANSAC to robustly estimate the camera parameters.
    ransac_parameters: Parameters for RANSAC. See
      _compute_homography_transform_ransac for valid keys.

  Returns:
    camera: jaxcam.Camera.
  """
  position = jnp.mean(rays.origins, axis=(0, 1))
  height, width, _ = rays.directions.shape
  identity_camera = jaxcam.Camera.create(
      orientation=jnp.eye(3),
      image_size=jnp.array([width, height]),
      position=jnp.zeros(3),
      focal_length=jnp.array([1.0]),
      principal_point=jnp.array([0, 0]),
  )
  identity_rays = get_rays_from_camera(identity_camera)
  intrinsics, orientation, _ = ray_dlt(
      identity_rays.directions.reshape(-1, 3),
      rays.directions.reshape(-1, 3),
      use_ransac=use_ransac,
      ransac_parameters=ransac_parameters,
  )
  focal_length = (intrinsics[0, 0] + intrinsics[1, 1]) / 2
  principal_point = intrinsics[:2, 2]
  skew = intrinsics[0, 1]
  return jaxcam.Camera.create(
      orientation=orientation,
      image_size=jnp.array([width, height]),
      position=position,
      focal_length=focal_length,
      principal_point=principal_point,
      skew=skew,
  )


def ray_dlt(
    directions1,
    directions2,
    use_ransac: bool = False,
    ransac_parameters: Optional[dict[str, Any]] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Solves for the optimal K and R transformation between two sets of rays.

  X2 ~ H @ X1 = R.T @ K_inv @ X1

  Args:
    directions1: (N, 3)
    directions2: (N, 3)
    use_ransac: If True, uses RANSAC to robustly estimate the camera parameters.
    ransac_parameters: Parameters for RANSAC. See
      _compute_homography_transform_ransac for valid keys.

  Returns:
    A tuple of (calibration, rotation, homography) where the calibration
    matrix (3, 3) is upper triangular, the rotation matrix (3, 3) is in
    world2camera convention, and the homography matrix (3, 3) is the optimal
    homography transform between the two sets of rays.
  """
  directions1 = directions1 / jnp.linalg.norm(
      directions1, axis=-1, keepdims=True
  )
  directions2 = directions2 / jnp.linalg.norm(
      directions2, axis=-1, keepdims=True
  )
  if use_ransac:
    if ransac_parameters is None:
      ransac_parameters = {}
    if 'rng' not in ransac_parameters:
      ransac_parameters['rng'] = jax.random.PRNGKey(0)
    homography = _compute_homography_transform_ransac(
        directions1=directions1,
        directions2=directions2,
        **ransac_parameters,
    )
  else:
    homography = _compute_homography_transform(directions1, directions2)
  out = jnp.linalg.qr(homography)
  # H = R.T @ K_inv, so R.T corresponds to the Q (orthonormal) and K_inv to the
  # R (upper triangular) in QR decomposition.
  intrinsics = jnp.linalg.inv(out.R)
  rotation = out.Q.T
  intrinsics /= intrinsics[2, 2]
  intrinsics, rotation = _positivize(intrinsics, rotation)
  return intrinsics, rotation, homography


@jax.jit
def _compute_homography_transform_ransac(
    rng,
    directions1: jnp.ndarray,
    directions2: jnp.ndarray,
    num_iterations: int = 200,
    num_correspondences: int = 8,
    threshold_deg: float = 1.0,
) -> jnp.ndarray:
  """Solves for the optimal homography transform with RANSAC.

  Args:
    rng: Random Number Generator.
    directions1: (N, 3) Unit normalized ray directions.
    directions2: (N, 3) Unit normalized ray directions.
    num_iterations: Number of iterations to run.
    num_correspondences: Number of correspondences to use per iteration.
    threshold_deg: Threshold (in degrees) for inliers.

  Returns:
    A (3, 3) homography matrix that has the most inliers.
  """

  def ransac_body(_, state):
    rng, best_inliers, best_homography = state
    rng, key = jax.random.split(rng)
    random_inds = jax.random.choice(
        key,
        jnp.arange(directions1.shape[0]),
        shape=(num_correspondences,),
        replace=False,
    )
    homography = _compute_homography_transform(
        directions1[random_inds], directions2[random_inds]
    )
    # Compute inliers by checking alignment of projected ray (opposite rays
    # also count as inliers).
    proj = math.matmul(homography, directions1.T).T
    proj = proj / jnp.linalg.norm(proj, keepdims=True, axis=-1)
    dot = jnp.clip(jnp.sum(proj * directions2, axis=-1), -1, 1)
    dist = jnp.arccos(jnp.abs(dot))
    is_inlier = dist * 180 / jnp.pi < threshold_deg
    num_inliers = jnp.sum(is_inlier)
    best_homography, best_inliers = jax.lax.cond(
        num_inliers > best_inliers,
        lambda _: (homography, num_inliers),
        lambda _: (best_homography, best_inliers),
        operand=None,
    )
    return rng, best_inliers, best_homography

  state = (rng, -1, jnp.eye(3))
  _, _, best_homography = jax.lax.fori_loop(
      0, num_iterations, ransac_body, state
  )
  return best_homography


@jax.jit
def _compute_homography_transform(
    directions1: jnp.ndarray,
    directions2: jnp.ndarray,
) -> jnp.ndarray:
  """Solves for the optimal homography transform between two sets of rays.

  Finds the homography transform H:
  x2 ~ H @ x1

  Args:
    directions1 (jnp.ndarray): (N, 3).
    directions2 (jnp.ndarray): (N, 3).

  Returns:
    A (3, 3) homography matrix.
  """
  x1, y1, z1 = jnp.split(directions1, 3, axis=-1)
  x2, y2, z2 = jnp.split(directions2, 3, axis=-1)
  z = jnp.zeros_like(x1)
  # Eq 4.1 in H&Z.
  a_x = jnp.concatenate(
      (z2 * x1, z2 * y1, z2 * z1, z, z, z, -x2 * x1, -x2 * y1, -x2 * z1), axis=1
  )
  a_y = jnp.concatenate(
      (z, z, z, -z2 * x1, -z2 * y1, -z2 * z1, y2 * x1, y2 * y1, y2 * z1), axis=1
  )
  homogeneous_matrix = jnp.concatenate((a_x, a_y), axis=0)
  _, _, vh = jnp.linalg.svd(homogeneous_matrix, full_matrices=False)
  homography = vh[-1]
  # Homography matrix is only defined up to a scale factor:
  homography = homography / jnp.linalg.norm(homography)
  homography = homography.reshape(3, 3)
  return homography


def _positivize(
    calib: jnp.ndarray, rotation: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Makes diagonal of calibration matrix positive.

  Ensures the diagonal entries of the calibration matrix (i.e. focal lengths)
  are positive, flipping the sign of the rows of the rotation matrix as
  necessary. Also ensures that the rotation matrix is valid (det(R) == 1).

  Args:
    calib: (3, 3) Calibration matrix.
    rotation: (3, 3) Rotation matrix.

  Returns:
    A tuple of (calib, rotation).
  """
  sign = jnp.sign(calib.diagonal())
  calib = calib * sign[None, :]
  rotation = rotation * sign[:, None]
  # Ensure valid rotation matrix.
  rotation = jax.lax.cond(
      jnp.linalg.det(rotation) < 0, lambda: -rotation, lambda: rotation
  )
  return calib, rotation
