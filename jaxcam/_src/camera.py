# Copyright 2023 The jaxcam Authors.
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

"""A JAX-based camera class."""

from collections.abc import Sequence
import dataclasses
import enum
import functools
from typing import Any, Generator, Optional, Union
import warnings

from flax import struct
import jax
import jax.numpy as jnp
from jaxcam._src import math
from numpy import typing as npt


def _camera_field(unbatched_shape: tuple[int, ...], **kwargs) -> ...:
  return dataclasses.field(
      metadata={
          'pytree_node': True,
          'unbatched_shape': unbatched_shape,
      },
      **kwargs,
  )


class ProjectionType(enum.Enum):
  """Camera projection type (perspective pinhole, or fisheye)."""

  PERSPECTIVE = 'perspective'
  FISHEYE = 'fisheye'


@struct.dataclass
class Camera:
  """The base class for a pinhole camera model.

  This class should not be instantiated directly. Use the `create()` function
  instead.

  References:
    https://en.wikipedia.org/wiki/Distortion_(optics)#Software_correction
    https://web.archive.org/web/20180312205006/https://www.asprs.org/wp-content/uploads/pers/1966journal/may/1966_may_444-462.pdf

  Attributes:
    orientation: [..., 3, 3] array of world-to-camera rotations.
    position: [..., 3] array of world-space positions.
    focal_length: [...] array or float of focal length in x.
    principal_point: [..., 2] array of the image-space principal point.
    image_size: [..., 2] array of image sizes.
    skew: [...] array or float of skew coefficients.
    pixel_aspect_ratio: [...] array or float of pixel aspect ratios (height /
      width)
    radial_distortion: [..., 4] array of radial distortion coefficients.
    tangential_distortion: [..., 2] array of tangential distortion coefficients.
    scale_factor_x: convenience property for focal_length.
    scale_factor_y: convenience property for focal_length * pixel_aspect_ratio.
    principal_point_x: convenience property for principal_point[..., 0].
    principal_point_y: convenience property for principal_point[..., 1].
    image_size_x: convenience property for image_size[..., 0].
    image_size_y: convenience property for image_size[..., 1].
    optical_axis: convenience property for orientation[..., 2, :].
    translation: the camera translation vector.
    world_to_camera_matrix: [..., 4, 4] the world-to-camera matrix of the camera
      (also known as the extrinsic matrix).
    intrinsic_matrix: [..., 3, 3] the intrinsic matrix of the camera.
    projection_type: the projection type of the camera.
    has_distortion: will be True if either radial or tangential distortion is
      enabled.
    has_radial_distortion: will be True if radial distortion is enabled.
    has_tangential_distortion: will be True if tangential distortion is enabled.
    use_inverted_distortion: if True, the meaning of distort/undistort is
      flipped.
  """
  orientation: jnp.ndarray = _camera_field((3, 3))
  position: jnp.ndarray = _camera_field((3,))
  focal_length: Union[jnp.ndarray, float] = _camera_field(())
  principal_point: jnp.ndarray = _camera_field((2,))
  image_size: jnp.ndarray = _camera_field((2,))
  skew: Union[jnp.ndarray, float] = _camera_field(())
  pixel_aspect_ratio: Union[jnp.ndarray, float] = _camera_field(())
  radial_distortion: jnp.ndarray | None = _camera_field((4,), default=None)
  tangential_distortion: jnp.ndarray | None = _camera_field((2,), default=None)

  # Non-pytree fields. For example, options to control camera logic. Logic is
  # baked in when jitting, so these fields do not need to be passed into jitted
  # functions and so can be left out of the pytree representation of the camera.
  projection_type: ProjectionType = struct.field(
      pytree_node=False, default=ProjectionType.PERSPECTIVE
  )
  use_inverted_distortion: bool = struct.field(pytree_node=False, default=False)

  @classmethod
  def create(
      cls,
      orientation: Optional[jnp.ndarray] = None,
      position: Optional[jnp.ndarray] = None,
      focal_length: Optional[jnp.ndarray] = None,
      principal_point: Optional[jnp.ndarray] = None,
      image_size: Optional[jnp.ndarray] = None,
      skew: Union[jnp.ndarray, float] = 0.0,
      pixel_aspect_ratio: Union[jnp.ndarray, float] = 1.0,
      radial_distortion: Optional[jnp.ndarray] = None,
      tangential_distortion: Optional[jnp.ndarray] = None,
      invert_distortion: bool = False,
      is_fisheye: bool = False,
  ) -> 'Camera':
    """Creates a camera with reasonable default values."""
    if position is None:
      position = jnp.zeros(3)
    if orientation is None:
      orientation = jnp.eye(3)
    if image_size is None:
      image_size = jnp.ones(2)
    if principal_point is None:
      principal_point = image_size / 2.0
    if focal_length is None:
      # Default focal length produces a FoV of 2*atan(0.5) ~= 53 degrees.
      focal_length = image_size[..., 0]

    # Ensure all items are strongly typed arrays to avoid triggering a cache
    # miss during JIT compilation due to weak type semantics.
    # See: https://jax.readthedocs.io/en/latest/type_promotion.html
    asarray = functools.partial(jnp.asarray, dtype=jnp.float32)

    kwargs = {
        'orientation': asarray(orientation),
        'position': asarray(position),
        'focal_length': asarray(focal_length),
        'principal_point': asarray(principal_point),
        'image_size': asarray(image_size),
        'skew': asarray(skew),
        'pixel_aspect_ratio': asarray(pixel_aspect_ratio),
    }

    if radial_distortion is not None:
      # Insert the 4th radial distortion coefficient if not present.
      radial_distortion = jnp.pad(
          asarray(radial_distortion),
          pad_width=(0, 4 - radial_distortion.shape[-1]),
      )
      kwargs['radial_distortion'] = asarray(radial_distortion)

    if tangential_distortion is not None:
      kwargs['tangential_distortion'] = asarray(tangential_distortion)

    if invert_distortion and (
        radial_distortion is not None or tangential_distortion is not None
    ):
      kwargs['use_inverted_distortion'] = True

    if is_fisheye:
      kwargs['projection_type'] = ProjectionType.FISHEYE
    else:
      kwargs['projection_type'] = ProjectionType.PERSPECTIVE

    return cls(**kwargs)

  def batch_axes(self) -> Optional['Camera']:
    """Returns the batch axis for each field.

    Camera fields may be individually batched or unbatched. This function
    inspects the field shapes and returns a Camera with an axis for each
    field: 0 if the field is batched, or None if the field is unbatched.
    The output of this function may be used as part of the in_axes parameter
    to jax.vmap. Using Camera with labels as fields is a bit of an abuse of the
    dataclass but jax.vmap requires that the structure of the input and in_axes
    matches exactly.

    Returns:
      a Camera with batch axis labels in each field, or None if no fields are
      batched.
    """
    fields = {f.name: f for f in self.pytree_fields()}
    field_is_batched = {
        k: len(jnp.array(v).shape) > len(fields[k].metadata['unbatched_shape'])
        for k, v in self.pytree_dict().items()
    }
    if not any(field_is_batched.values()):
      return None
    field_axes = {k: 0 if v else None for k, v in field_is_batched.items()}
    return type(self)(**field_axes)

  @classmethod
  def pytree_fields(cls) -> list[dataclasses.Field[Any]]:
    fields = dataclasses.fields(cls)
    return [f for f in fields if f.metadata['pytree_node']]

  def pytree_dict(self) -> dict[str, Any]:
    """Returns a dict of the dataclass containing only pytree fields.

    Returns:
      A dictionary containing enabled pytree fields. None-type fields will be
      omitted.
    """
    pytree_fields = self.pytree_fields()
    return {
        field.name: self.__dict__[field.name]
        for field in pytree_fields
        if self.__dict__[field.name] is not None
    }

  @property
  def ndim(self) -> int:
    """Returns the number of batch dimensions."""
    return len(self.shape)

  @property
  def shape(self) -> tuple[int, ...]:
    """The shape of the batch dimensions of the camera."""
    batch_axes = self.batch_axes()
    if batch_axes is None:
      return ()

    shape = ()
    # This class assumes that the batch dimensions of all batched Attributes
    # are the same, so we can just return the shape of any such attribute.
    for field in dataclasses.fields(self):
      batch_axis = getattr(batch_axes, field.name)
      if batch_axis is not None:
        array = getattr(self, field.name)
        unbatched_ndim = len(field.metadata['unbatched_shape'])
        batch_ndim = array.ndim - unbatched_ndim
        shape = array.shape[:batch_ndim]
        break

    return shape

  def __len__(self) -> int:
    """Return the number of cameras in the batch.

    Raises:
      TypeError: If the camera is unbatched.

    Returns:
      The number of cameras in the batch. Follows the same convention as
        numpy as returns the first dimension of the shape if multiple batch
        dimensions exist.
    """
    if self.ndim == 0:
      raise TypeError('len() of unbatched camera')
    return self.shape[0]

  def __getitem__(self, index: npt.ArrayLike) -> 'Camera':
    """Convenience method for slicing the camera.

    Slices the camera. Any slicing object that works with numpy arrays will
    work here.

    All batched fields must have the same batch dimension and unbatched fields
    will remain unbatched.

    Examples:

    ```python
    # Get the first 10 cameras.
    cameras[:10]

    # Get the 5th camera.
    cameras[5]

    # Index with an array.
    cameras[jnp.ndarray([0, 2, 4, 6])]
    ```

    Args:
      index: The slicing object.

    Raises:
      IndexError: If called on an unbatched camera.

    Returns:
      The sliced camera.
    """
    batch_axes = self.batch_axes()
    if batch_axes is None:
      raise IndexError('Cannot slice unbatched camera.')

    kwargs = {}
    pytree_dict = self.pytree_dict()
    for k, v in self.__dict__.items():
      if k in pytree_dict and v is not None:
        kwargs[k] = v[index] if getattr(batch_axes, k) is not None else v
      else:
        kwargs[k] = v

    return self.__class__(**kwargs)

  def __iter__(self) -> Generator['Camera', None, None]:
    """Iterate over the camera batch.

    Iterates along the first batch dimension of the camera. For example, the
    following are equivalent:

      ```python
      [cam for cam in camera]
      [camera[i] for i in len(camera)]
      ```

    Raises:
      IndexError: If called on an unbatched camera.

    Yields:
      Cameras with each element sliced along the first batch dimension.
    """
    if self.ndim == 0:
      raise TypeError('iteration over a 0-d camera')
    for i in range(len(self)):
      yield self[i]

  @property
  def scale_factor_x(self) -> jnp.ndarray:
    return self.focal_length  # pytype: disable=bad-return-type  # jax-ndarray

  @property
  def scale_factor_y(self) -> jnp.ndarray:
    return self.focal_length * self.pixel_aspect_ratio  # pytype: disable=bad-return-type  # jax-ndarray

  @property
  def principal_point_x(self) -> jnp.ndarray:
    return self.principal_point[..., 0]

  @property
  def principal_point_y(self) -> jnp.ndarray:
    return self.principal_point[..., 1]

  @property
  def image_size_x(self) -> jnp.ndarray:
    return self.image_size[..., 0]

  @property
  def image_size_y(self) -> jnp.ndarray:
    return self.image_size[..., 1]

  @property
  def optical_axis(self) -> jnp.ndarray:
    return self.orientation[..., 2, :]

  @property
  def has_distortion(self) -> bool:
    return self.has_radial_distortion or self.has_tangential_distortion

  @property
  def has_radial_distortion(self) -> bool:
    return self.radial_distortion is not None

  @property
  def has_tangential_distortion(self) -> bool:
    return self.tangential_distortion is not None

  @property
  def translation(self) -> jnp.ndarray:
    # pylint: disable=invalid-unary-operand-type
    return math.matmul(-self.orientation, self.position[..., None]).squeeze(-1)

  @property
  def world_to_camera_matrix(self) -> jnp.ndarray:
    """Returns the 4x4 matrix that takes world points to camera coordinates."""
    matrix = jnp.empty((*self.shape, 3, 4))
    matrix = matrix.at[..., :3, :3].set(self.orientation)
    matrix = matrix.at[..., :3, 3].set(self.translation)
    return _to_matrix_4x4(matrix)

  @property
  def camera_to_world_matrix(self) -> jnp.ndarray:
    """Returns the 4x4 matrix that takes camera points to world coordinates."""
    matrix = jnp.empty((*self.shape, 3, 4))
    matrix = matrix.at[..., :3, :3].set(self.orientation.transpose(-1, -2))
    matrix = matrix.at[..., :3, 3].set(self.position)
    return _to_matrix_4x4(matrix)

  @property
  def intrinsic_matrix(self) -> jnp.ndarray:
    """Returns the intrinsic matrix that maps local coordinates to pixels."""
    matrix = jnp.zeros((*self.shape, 3, 3))
    matrix = matrix.at[..., 0, 0].set(self.scale_factor_x)
    matrix = matrix.at[..., 0, 1].set(self.skew)
    matrix = matrix.at[..., 0, 2].set(self.principal_point_x)
    matrix = matrix.at[..., 1, 1].set(self.scale_factor_y)
    matrix = matrix.at[..., 1, 2].set(self.principal_point_y)
    matrix = matrix.at[..., 2, 2].set(1)
    return matrix


def create(*args, **kwargs) -> Camera:
  warnings.warn(
      'The `create()` function has been moved to `Camera.create()`.'
      '`create()` will be removed in the future.',
      DeprecationWarning,
  )
  return Camera.create(*args, **kwargs)


def _to_matrix_4x4(matrix):
  """Converts a matrix to a 4x4 matrix."""
  if matrix.shape[-2] > 4 or matrix.shape[-1] > 4:
    raise ValueError('Matrix dimensions must be a maximum of 4x4.')

  num_rows, num_cols = matrix.shape[-2:]
  matrix_4x4 = jnp.zeros((*matrix.shape[:-2], 4, 4))
  matrix_4x4 = matrix_4x4.at[..., 3, 3].set(1)
  matrix_4x4 = matrix_4x4.at[..., :num_rows, :num_cols].set(matrix)
  return matrix_4x4


def _distort_local_pixels(
    camera: Camera, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Distorts normalized image pixels."""
  if camera.use_inverted_distortion:
    x, y = _radial_and_tangential_undistort(
        x,
        y,
        radial_distortion=camera.radial_distortion,
        tangential_distortion=camera.tangential_distortion,
    )
  elif camera.has_distortion:
    x, y = _radial_and_tangential_distort(
        x,
        y,
        radial_distortion=camera.radial_distortion,
        tangential_distortion=camera.tangential_distortion,
    )

  return x, y


def _undistort_local_pixels(
    camera: Camera, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Undistorts normalized image pixels."""
  if camera.use_inverted_distortion:
    x, y = _radial_and_tangential_distort(
        x,
        y,
        radial_distortion=camera.radial_distortion,
        tangential_distortion=camera.tangential_distortion,
    )
  elif camera.has_distortion:
    x, y = _radial_and_tangential_undistort(
        x,
        y,
        radial_distortion=camera.radial_distortion,
        tangential_distortion=camera.tangential_distortion,
    )

  return x, y


def pixels_to_local_rays(
    camera: Camera, pixels: jnp.ndarray, normalize: bool = True
) -> jnp.ndarray:
  """Returns local ray directions for the provided pixels."""
  y = ((pixels[..., 1] - camera.principal_point_y) / camera.scale_factor_y)
  x = (
      pixels[..., 0] - camera.principal_point_x - y * camera.skew
  ) / camera.scale_factor_x

  x, y = _undistort_local_pixels(camera, x, y)

  dirs = jnp.stack([x, y, jnp.ones_like(x)], axis=-1)
  if camera.projection_type is ProjectionType.FISHEYE:
    theta = jnp.sqrt(x ** 2 + y ** 2)
    theta = jnp.minimum(jnp.pi, theta)
    sin_theta_over_theta = jnp.sin(theta) / theta
    fisheye_dir = jnp.stack([
        dirs[..., 0] * sin_theta_over_theta,
        dirs[..., 1] * sin_theta_over_theta,
        jnp.cos(theta),
    ], axis=-1)
    eps = jnp.finfo(x.dtype).eps
    # It is approximately perspective when the viewing ray is too close to the
    # optical axis.
    if normalize:
      dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = jnp.where(theta[..., None] < eps, dirs, fisheye_dir)
  else:
    if normalize:
      dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

  return dirs


def pixels_to_rays(
    camera: Camera, pixels: jnp.ndarray, normalize: bool = True
) -> jnp.ndarray:
  """Returns world-space rays for the provided pixels.

  Args:
    camera: Camera instance.
    pixels: [..., 2] array containing 2d pixel positions.
    normalize: If True, normalize the output ray directions. If False, the
      returned directions will be points at the image plane at distance 1 from
      camera center. Defaults to True.

  Returns:
      An array containing the normalized ray directions in world coordinates.
  """
  if pixels.shape[-1] != 2:
    raise ValueError('The last dimension of pixels must be 2.')

  batch_shape = pixels.shape[:-1]
  pixels = jnp.reshape(pixels, (-1, 2))

  local_rays_dir = pixels_to_local_rays(camera, pixels, normalize=normalize)
  rays_dir = math.matmul(
      jnp.swapaxes(camera.orientation, -1, -2), local_rays_dir[..., jnp.newaxis]
  )
  rays_dir = jnp.squeeze(rays_dir, axis=-1)
  rays_dir = rays_dir.reshape((*batch_shape, 3))
  return rays_dir


def pixels_to_points(
    camera: Camera, pixels: jnp.ndarray, depth: jnp.ndarray
) -> jnp.ndarray:
  """Unprojects pixels and depth to (x,y,z,w) homogenous world-space points."""
  rays_through_pixels = pixels_to_rays(camera, pixels)
  ray_depth = depth_to_ray_depth(camera, rays_through_pixels, depth)
  points = rays_through_pixels * ray_depth[..., None] + camera.position
  pad_shape = [(0, 0)] * len(depth.shape) + [(0, 1)]
  points = jnp.pad(points, pad_shape, constant_values=1.0)
  return points


def world_points_to_local_points(camera: Camera,
                                 world_points: jnp.ndarray) -> jnp.ndarray:
  """Transforms world-space (x,y,z) points to local-space (x,y,z) points.

  Local-space coordinates are also known as camera-space coordinates.

  Args:
    camera: The camera defining the local coordinate frame.
    world_points: (..., 3) The world-space points to transform.

  Returns:
    The local-space coordinates.
  """
  world_points_flat = world_points.reshape((-1, 3))
  translated_points = world_points_flat - camera.position
  local_points = (math.matmul(camera.orientation, translated_points.T)).T
  return local_points.reshape(world_points.shape)


def local_points_to_world_points(
    camera: Camera, local_points: jnp.ndarray
) -> jnp.ndarray:
  """Transforms local-space (x,y,z) points to world-space (x,y,z) points.

  Local-space coordinates are also known as camera-space coordinates.

  Args:
    camera: The camera defining the local coordinate frame.
    local_points: (..., 3) The local-space points to transform.

  Returns:
    The world-space coordinates.
  """
  local_points_flat = local_points.reshape((-1, 3))

  rotated_points = math.matmul(camera.orientation.T, local_points_flat.T).T
  world_points = rotated_points + camera.position
  return world_points.reshape(world_points.shape)


def depth_to_ray_depth(camera: Camera, ray: jnp.ndarray,
                       depth: jnp.ndarray) -> jnp.ndarray:
  """Converts depth along the optical axis to depth along ray.

  Args:
    camera: The camera defining the local coordinate frame.
    ray: (..., 3) The ray corresponding to each point in `depth`.
    depth: (...,) The depth along the optical axis for each ray in `ray`.

  Returns:
    (..., 3) The depth along the optical axis for each ray.
  """
  cosa = math.matmul(ray, camera.optical_axis)
  return depth / cosa


def ray_depth_to_depth(camera: Camera, ray: jnp.ndarray,
                       ray_depth: jnp.ndarray) -> jnp.ndarray:
  """Converts depth along ray to depth along the optical axis.

  Args:
    camera: The camera defining the local coordinate frame.
    ray: (..., 3) The ray corresponding to each point in `ray_depth`.
    ray_depth: (...,) The depth along the ray for each ray in `ray`.

  Returns:
    (..., 3) The depth along the optical axis for each ray.
  """
  cosa = math.matmul(ray, camera.optical_axis)
  return ray_depth * cosa


def world_to_camera_matrix(camera: Camera) -> jnp.ndarray:
  """Returns the 4x4 matrix that takes world points to camera coordinates.

  DEPRECATED: Use Camera.world_to_camera_matrix.

  Args:
    camera: The camera to query.

  Returns:
    The world-to-camera matrix.
  """
  return camera.world_to_camera_matrix


def update_world_to_camera_matrix(
    camera: Camera, matrix: jnp.ndarray
) -> Camera:
  """Sets the transform that takes world points to camera coordinates.

  Args:
    camera: Camera to be used to set camera_from_world transform.
    matrix: A (..., 4, 4) jax array containing a homogeneous transform, where
      the last row is [0, 0, 0, 1].

  Returns:
    Camera with updated orientation and position after applying transform.
  """

  if matrix.shape[-2:] != (4, 4):
    raise ValueError('Last two axes of transform must have shape (3, 3).')

  orientation = matrix[..., :3, :3]
  translation = matrix[..., :3, 3]
  position = math.einsum('... j, ... j k -> ... k', translation, -orientation)
  return camera.replace(position=position, orientation=orientation)


def update_translation(camera: Camera, translation: jnp.ndarray) -> Camera:
  """Updates the translation of the camera.

  Args:
    camera: Camera to set the translation for.
    translation: A (..., 3) array containing the new translation.

  Returns:
    Camera with updated translation.
  """
  new_position = math.einsum(
      '... j, ... j k -> ... k', translation, -camera.orientation
  )
  return camera.replace(position=new_position)


def update_intrinsic_matrix(
    camera: Camera, intrinsic_matrix: jnp.ndarray
) -> Camera:
  """Sets the camera intrinsics based on an intrinsic matrix.

  Args:
    camera: Camera to set the intrinsics for.
    intrinsic_matrix: A (..., 3, 3) jax array containing an intrinsic atrix.

  Returns:
    Camera with updated intrinsics.
  """
  if intrinsic_matrix.shape[-2:] != (3, 3):
    raise ValueError(
        'Last two axes of intrinsic_matrix must have shape (3, 3).'
    )

  focal_length_x = intrinsic_matrix[..., 0, 0]
  focal_length_y = intrinsic_matrix[..., 1, 1]
  pixel_aspect_ratio = focal_length_y / focal_length_x
  principal_point_x = intrinsic_matrix[..., 0, 2]
  principal_point_y = intrinsic_matrix[..., 1, 2]
  return camera.replace(
      focal_length=focal_length_x,
      pixel_aspect_ratio=pixel_aspect_ratio,
      skew=intrinsic_matrix[..., 0, 1],
      principal_point=jnp.stack(
          [principal_point_x, principal_point_y], axis=-1
      ),
  )


def invert_radial_distortion_coefficients(
    radial_distortion: jnp.ndarray,
) -> jnp.ndarray:
  """Inverts radial distortion parameters.

  This uses the exact inverse formula from Drap and Lefèvre (Sensors, 2016).

  Reference:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4934233/

  Args:
    radial_distortion: The input radial distortion parameters.

  Returns:
    The inverted radial distortion parameters.
  """
  a1, a2, a3, a4 = jnp.split(radial_distortion, 4, axis=-1)
  b1 = -a1
  b2 = 3 * a1**2 - a2
  b3 = 8 * a1 * a2 - 12 * a1**3 - a3
  b4 = 55 * a1**4 + 10 * a1 * a3 - 55 * a1**2 * a2 + 5 * a2**2 - a4
  return jnp.concatenate([b1, b2, b3, b4], axis=-1)


def _radial_and_tangential_distort(
    x: jnp.ndarray,
    y: jnp.ndarray,
    radial_distortion: jnp.ndarray | None = None,
    tangential_distortion: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the distorted pixel positions."""
  dx_radial, dy_radial = 0.0, 0.0
  dx_tangential, dy_tangential = 0.0, 0.0
  r2 = x * x + y * y

  if radial_distortion is not None:
    k1, k2, k3, k4 = radial_distortion
    radial_distortion = r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
    dx_radial = x * radial_distortion
    dy_radial = y * radial_distortion

  if tangential_distortion is not None:
    p1, p2 = tangential_distortion
    dx_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy_tangential = 2 * p2 * x * y + p1 * (r2 + 2 * y * y)

  return x + dx_radial + dx_tangential, y + dy_radial + dy_tangential


def _radial_and_tangential_undistort(
    x_distorted: jnp.ndarray,
    y_distorted: jnp.ndarray,
    radial_distortion: jnp.ndarray,
    tangential_distortion: jnp.ndarray,
    max_iterations: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Undistorts a point in image-space using an iterative optimization.

  This uses Newton-Raphson (related to Gauss-Newton) to find the undistorted
  pixel positiong by finding the roots of the equation `distort(x) - x_d = 0`.

  References:
    https://en.wikipedia.org/wiki/Newton%27s_method#k_variables,_k_functions

  Args:
    x_distorted: The distorted x pixel position.
    y_distorted: The distorted y pixel position.
    radial_distortion: (4,) The radial distortion parameters.
    tangential_distortion: (2,) The tangential distortion parameters.
    max_iterations: The number of iterations to run Newton's method.

  Returns:
    A tuple containing the undistorted x and y positions.
  """

  def f(xy, xy_distorted):
    x_distorted_pred, y_distorted_pred = _radial_and_tangential_distort(
        xy[..., 0],
        xy[..., 1],
        radial_distortion=radial_distortion,
        tangential_distortion=tangential_distortion,
    )
    xy_distorted_pred = jnp.stack([x_distorted_pred, y_distorted_pred], axis=-1)
    return xy_distorted_pred - xy_distorted

  batch_shape = x_distorted.shape
  xy_distorted = jnp.stack([x_distorted, y_distorted], axis=-1)
  xy_distorted = xy_distorted.reshape(-1, 2)

  # Run Newton's method.
  jac_fn = jax.jacfwd(f)
  xy = xy_distorted
  for _ in range(max_iterations):
    jac = jax.vmap(jac_fn)(xy, xy_distorted)
    # Solve the linear system: J(x_n)(x{n+1}-x_n) = -f(x_n).
    # residual = x_{n+1} - x_n, so x_{n+1} = residual + x_n.
    residual = jax.vmap(jnp.linalg.solve)(jac, -f(xy, xy_distorted))
    xy = residual + xy

  # Reshape back to the original shape.
  xy = xy.reshape(*batch_shape, 2)
  return xy[..., 0], xy[..., 1]


@jax.jit
def project(camera: Camera, points: jnp.ndarray) -> jnp.ndarray:
  """Projects world-space 3D points (x,y,z) to pixel positions (x,y).

  Args:
    camera: The camera to project with.
    points: (..., 3) The 3D world-space points to project.

  Returns:
    A (..., 2) array containing the projected pixel locations.
  """
  eps = jnp.finfo(points.dtype).eps
  batch_shape = points.shape[:-1]
  local_points = world_points_to_local_points(camera, points)

  local_x = local_points[..., 0]
  local_y = local_points[..., 1]
  local_z = local_points[..., 2]

  # Get normalized local pixel positions.
  x = local_x / local_z
  y = local_y / local_z
  if camera.projection_type is ProjectionType.FISHEYE:
    r = jnp.sqrt(local_x ** 2 + local_y ** 2)
    theta = jnp.arctan2(r, local_z)
    fisheye_x = theta / r * local_x
    fisheye_y = theta / r * local_y
    # When theta is small it is approximately the same as perspective.
    x = jnp.where(theta < eps, x, fisheye_x)
    y = jnp.where(theta < eps, y, fisheye_y)

  x, y = _distort_local_pixels(camera, x, y)

  # Map the distorted ray to the image plane and return the depth.
  pixel_x = camera.focal_length * x + camera.skew * y + camera.principal_point_x
  pixel_y = (
      camera.focal_length * camera.pixel_aspect_ratio * y +
      camera.principal_point_y)

  pixels = jnp.stack([pixel_x, pixel_y], axis=-1)
  return pixels.reshape((*batch_shape, 2))


def look_at(
    camera: Camera,
    eye: jnp.ndarray,
    center: jnp.ndarray,
    world_up: jnp.ndarray,
    camera_convention: str = 'opencv',
) -> Camera:
  """Applies a look-at transform to the given camera.

  Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

  Args:
    camera: The camera to move.
    eye: (3,) The position of the camera.
    center: (3,) The position the camera will "look at".
    world_up: (3,) The world's up direction.
    camera_convention: one of 'opengl' or 'opencv'. In 'opengl' convention, the
      camera aims down the -Z axis and the viewport origin is lower-left. In
      'opencv' convention, the camera aims down the +Z axis and the viewport
      origin is upper-left.

  Returns:
    A copy of the camera but with the look-at transform applied.
  """
  # NOTE: The computation before conversion happens in OpenGL coordinates.
  vector_degeneracy_cutoff = 1e-6
  forward = center - eye
  forward_norm = jnp.linalg.norm(forward)

  try:
    if forward_norm < vector_degeneracy_cutoff:
      raise ValueError(
          'Camera matrix is degenerate because eye and center are close.'
      )
  except jax.errors.ConcretizationTypeError:
    pass

  forward = forward / forward_norm

  to_side = jnp.cross(forward, world_up)
  to_side_norm = jnp.linalg.norm(to_side)
  try:
    if to_side_norm < vector_degeneracy_cutoff:
      raise ValueError(
          'Camera matrix is degenerate because up and gaze are close '
          'or up is degenerate.'
      )
  except jax.errors.ConcretizationTypeError:
    pass

  to_side = to_side / to_side_norm
  cam_up = jnp.cross(to_side, forward)

  # Make a 3x3 rotation matrix:
  view_rotation = jnp.vstack([to_side, cam_up, -forward])
  # Set the upper 3x3 of a 4x4 identity matrix:
  view_rotation = jnp.eye(4).at[:3, :3].set(view_rotation)

  # Make a 4x4 translation matrix:
  view_translation = jnp.eye(4).at[:3, 3].set(-eye)

  cam_from_world = math.matmul(view_rotation, view_translation)

  if camera_convention == 'opencv':
    # Equal to left multiplication by np.diag([1.0, -1.0, -1.0, 1.0])
    cam_from_world = cam_from_world.at[1:3, :].multiply(-1.0)
  elif camera_convention != 'opengl':
    raise ValueError(f'Unknown camera convention {camera_convention}')

  return update_world_to_camera_matrix(camera, cam_from_world)


def get_pixel_centers(image_width: int, image_height: int) -> jnp.ndarray:
  """Returns the pixel centers."""
  xx, yy = jnp.meshgrid(
      jnp.arange(image_width, dtype=jnp.float32),
      jnp.arange(image_height, dtype=jnp.float32))
  return jnp.stack([xx, yy], axis=-1) + 0.5


def scale(camera: Camera, amount: float | jnp.ndarray) -> Camera:
  """Scales the camera by the given amount."""
  # First round the image size to a round number.
  new_image_size = jnp.round(camera.image_size * amount)
  return scale_to_image_size(camera, new_image_size)


def scale_to_image_size(camera: Camera, image_size: jnp.ndarray) -> Camera:
  """Scales the camera to the given image size."""
  scale_xy = image_size / camera.image_size
  scale_x = scale_xy[..., 0]
  scale_y = scale_xy[..., 1]
  return camera.replace(
      focal_length=camera.focal_length * scale_x,
      pixel_aspect_ratio=scale_y / scale_x * camera.pixel_aspect_ratio,
      principal_point=camera.principal_point * scale_xy,
      skew=camera.skew * scale_x,
      image_size=image_size,
  )


def crop(
    camera: Camera,
    left: int | jnp.ndarray = 0,
    right: int | jnp.ndarray = 0,
    top: int | jnp.ndarray = 0,
    bottom: int | jnp.ndarray = 0,
) -> Camera:
  """Returns a copy of the camera with adjusted image bounds.

  NOTE: This function does not perform bounds checking.

  Args:
    camera: Input camera.
    left: Number of pixels by which to reduce (or augment, if negative) the left
      boundary of the image domain extent.
    right: Number of cropped pixels for the right image boundary.
    top: Number of cropped pixels for the top image boundary.
    bottom: Number of cropped pixels for the bottom image boundary.

  Returns:
    A camera with adjusted image dimensions. The focal length is unchanged, and
    the principal point is updated to preserve the original principal axis.
  """
  crop_left_top = jnp.array([left, top])
  crop_right_bottom = jnp.array([right, bottom])
  new_image_size = camera.image_size - crop_left_top - crop_right_bottom
  new_principal_point = camera.principal_point - crop_left_top

  return camera.replace(
      principal_point=new_principal_point,
      image_size=new_image_size,
  )


def concatenate(cameras: Sequence[Camera], axis: int = 0) -> Camera:
  """Concatenates two batched cameras.

  This function operates primarily on _batched_ cameras. Either all cameras
  must be batched, or all cameras must be unbatched. In the latter case a single
  batch dimension will be created, similar to when concatenating scalars in
  numpy.

  If some attributes are unbatched, they will be expanded and broadcasted to the
  batch shape.

  Args:
    cameras: A list of cameras to concatenate.
    axis: The batch axis to concatenate across.

  Raises:
    ValueError: If no cameras are given.
    TypeError: If the batch axes do not match.

  Returns:
    The concatenated camera.
  """

  if not cameras:
    raise ValueError('Need at least one camera to concatenate.')

  if not all(c.ndim == cameras[0].ndim for c in cameras):
    dimensions = ', '.join([str(c.shape) for c in cameras])
    raise TypeError(
        f'Cannot concatenate cameras with different numbers of dimensions'
        f': got {dimensions}')

  # Expand unbatched cameras.
  cameras = [(c if c.shape else jax.tree_map(
      functools.partial(jnp.expand_dims, axis=0), c)) for c in cameras]

  cls = cameras[0].__class__
  kwargs = {}
  for field in dataclasses.fields(cls):
    # For non-field values pass the value of the first camera assuming they are
    # all the same.
    first_value = getattr(cameras[0], field.name)
    if not field.metadata['pytree_node'] or first_value is None:
      kwargs[field.name] = first_value
      continue

    unbatched_shape = field.metadata['unbatched_shape']

    def _expand_and_broadcast(c):
      # pylint: disable=cell-var-from-loop
      v = jnp.asarray(getattr(c, field.name))
      ndim = v.ndim - len(unbatched_shape)
      v = jnp.expand_dims(v, axis=tuple(range(len(c.shape) - ndim)))
      v = jnp.broadcast_to(v, shape=(*c.shape, *unbatched_shape))
      return v

    values = [_expand_and_broadcast(c) for c in cameras]
    kwargs[field.name] = jnp.concatenate(values, axis=axis)

  return cls(**kwargs)


def transform(
    camera: Camera,
    scale: jnp.ndarray,  # pylint: disable=redefined-outer-name
    rotation: jnp.ndarray,
    translation: jnp.ndarray,
) -> Camera:
  """Applies a similarity transform to the camera.

  The scale and rotation are applied first followed by the translation. I.e.,
  if the translation is in the input coordinates, it should be rotated and
  scaled prior to being passed to this function.

  The new position becomes `scale * rotation @ position + translation`.
  The new orientation becomes `orientation @ rotation.T`.

  Args:
    camera: The camera to transform.
    scale: (1,) The scale of the transform.
    rotation: (3, 3) The rotation of the transform.
    translation: (3,) The translation of the transform.

  Returns:
    The transformed camera.
  """
  new_orientation = math.matmul(camera.orientation, rotation.T)
  new_position = math.transform_point(
      camera.position, scale=scale, rotation=rotation, translation=translation
  )
  camera = camera.replace(orientation=new_orientation, position=new_position)
  return camera


def relative_transform(reference: Camera, target: Camera) -> jnp.ndarray:
  """Computes the transform from reference camera to the target camera.

  The transform relativizes the camera such that the target cameras are in the
  reference camera's reference frame. I.e., the reference camera's camera
  coordinate frame becomes the world coordinate frame.

  Args:
    reference: The reference camera.
    target: The target camera.

  Returns:
    The relative transform between the target and reference camera. This matrix
    takes points in the reference camera's coordinates and transforms them to
    be in the target camera's coordinates.
  """
  return math.matmul(
      target.world_to_camera_matrix, reference.camera_to_world_matrix
  )


def essential_matrix(reference: Camera, target: Camera) -> jnp.ndarray:
  """Computes the essential matrix between two cameras.

  The essential matrix will be computed based on the transformation from the
  reference to the target camera.

  References:
    https://en.wikipedia.org/wiki/Essential_matrix

  Args:
    reference: The reference camera.
    target: The target camera.

  Returns:
    A (3, 3) matrix containing the essential matrix.
  """
  rel_transform = relative_transform(reference, target)
  rel_target = update_world_to_camera_matrix(target, rel_transform)
  return math.matmul(math.skew(rel_target.translation), rel_target.orientation)


def fundamental_matrix(reference: Camera, target: Camera) -> jnp.ndarray:
  """Computes the fundamental matrix between two cameras.

  References:
    https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)

  Args:
    reference: The reference camera.
    target: The target camera.

  Returns:
    A (3, 3) matrix containing the fundamental matrix. When this matrix is left
    multiplied to a homogeneous pixel coordinate in the reference camera, it
    results in a homogeneous line equation containing the corresponding epipolar
    line in the target camera.
  """
  essential_mat = essential_matrix(reference, target)
  left = jnp.linalg.inv(target.intrinsic_matrix).T
  right = jnp.linalg.inv(reference.intrinsic_matrix)
  return math.matmul(left, math.matmul(essential_mat, right))
