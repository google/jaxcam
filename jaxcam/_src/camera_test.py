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

"""Unit tests for jaxcam."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import random
import jaxcam
from jaxcam._src import math
import numpy as np


_IMAGE_WIDTH = 512
_IMAGE_HEIGHT = 512
_FOCAL_LENGTH = 512
_SAMPLE_ROTATION = np.array([
    [0.22629565, -0.18300793, 0.95671225],
    [0.95671225, 0.22629565, -0.18300793],
    [-0.18300793, 0.95671225, 0.22629565],
])
_SAMPLE_POINTS = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
])

# Since Jaxcam functions generally use up to 3 inputs, we define a collection of
# triplets of batch shapes that broadcast.
_SHAPES_THAT_BROADCAST = (
    ((), (), ()),
    ((7,), (7,), (7,)),
    ((1,), (7,), (7,)),
    ((7,), (1,), (7,)),
    ((7,), (7,), (1,)),
    ((11, 7), (11, 7), (11, 7)),
    ((13,), (11, 13), (7, 11, 13)),
    ((7, 11, 13), (11, 13), (13,)),
    ((11, 1), (1, 11, 13), (7, 1, 13)),
    ((), (11,), (7, 11)),
    ((7, 11), (11,), ()),
    ((1,), (1,), (1,)),
    ((), (1,), (1, 1)),
    ((1, 1), (1,), ()),
)


def random_camera(
    rng,
    radius: int = 1,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
):
  rng, key = random.split(rng)
  position = random.normal(key, (3,))
  position = radius * position / jnp.linalg.norm(position)
  camera = jaxcam.Camera.create(
      focal_length=jnp.array(_FOCAL_LENGTH),
      image_size=jnp.array((_IMAGE_WIDTH, _IMAGE_HEIGHT)),
  )
  camera = jaxcam.look_at(
      camera, eye=position, center=jnp.array(look_at), world_up=jnp.array(up)
  )
  return camera


class CameraTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = random.PRNGKey(0)

  def test_world_to_camera_is_inverse_of_camera_to_world_matrix(self):
    camera = random_camera(self.rng)

    np.testing.assert_allclose(
        camera.world_to_camera_matrix,
        jnp.linalg.inv(camera.camera_to_world_matrix),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        camera.camera_to_world_matrix,
        jnp.linalg.inv(camera.world_to_camera_matrix),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        camera.world_to_camera_matrix @ camera.camera_to_world_matrix,
        jnp.eye(4),
        atol=1e-6,
    )

  @parameterized.product(
      scale=[
          0.1,
          1.0,
          2.0,
      ],
      rotation=[
          np.eye(3),
          np.array([
              [1.0, 0.0, 0.0],
              [0.0, -0.41614687, -0.9092974],
              [0.0, 0.9092974, -0.41614687],
          ]),
          np.array([
              [0.7071067, 0.0, 0.7071068],
              [0.0, 1.0, 0.0],
              [-0.7071068, 0.0, 0.7071067],
          ]),
          np.array([
              [0.7071067, -0.7071068, 0.0],
              [0.7071068, 0.7071067, 0.0],
              [0.0, 0.0, 1.0],
          ]),
          np.array([
              [0.22629565, -0.18300793, 0.95671225],
              [0.95671225, 0.22629565, -0.18300793],
              [-0.18300793, 0.95671225, 0.22629565],
          ]),
      ],
      translation=[
          np.zeros(3),
          np.ones(3),
          np.array([0.1, 0.0, -0.1]),
      ],
  )
  def test_transform_projection_identical(self, scale, rotation, translation):
    camera = random_camera(self.rng)
    points = random.uniform(self.rng, (100, 3), minval=-0.2, maxval=0.2)
    projected_points = jaxcam.project(camera, points)
    transformed_camera = jaxcam.transform(camera, scale, rotation, translation)
    transformed_points = jax.vmap(
        math.transform_point, in_axes=(0, None, None, None)
    )(points, scale, rotation, translation)
    transformed_projected_points = jaxcam.project(
        transformed_camera, transformed_points
    )

    np.testing.assert_allclose(
        projected_points, transformed_projected_points, atol=1e-3
    )

  def test_world_points_to_local_points_regression(self):
    camera = jaxcam.Camera.create(
        orientation=_SAMPLE_ROTATION,
        position=jnp.array([0.5, 1.5, 2.5]),
    )

    # Since there are no proper regression tests, the old code is copied here.
    def world_points_to_local_points_old(
        camera: jaxcam.Camera,
        world_points: jnp.ndarray,
    ) -> jnp.ndarray:
      world_points_flat = world_points.reshape((-1, 3))
      translated_points = world_points_flat - camera.position
      local_points = (math.matmul(camera.orientation, translated_points.T)).T
      return local_points.reshape(world_points.shape)

    np.testing.assert_allclose(
        world_points_to_local_points_old(camera, _SAMPLE_POINTS),
        jaxcam.world_points_to_local_points(camera, _SAMPLE_POINTS),
    )

  def test_local_points_to_world_points_regression(self):
    camera = jaxcam.Camera.create(
        orientation=_SAMPLE_ROTATION,
        position=jnp.array([0.5, 1.5, 2.5]),
    )

    # Since there are no proper regression tests, the old code is copied here.
    def local_points_to_world_points_old(
        camera: jaxcam.Camera,
        local_points: jnp.ndarray
    ) -> jnp.ndarray:
      local_points_flat = local_points.reshape((-1, 3))
      rotated_points = math.matmul(camera.orientation.T, local_points_flat.T).T
      world_points = rotated_points + camera.position
      return world_points.reshape(local_points.shape)

    np.testing.assert_allclose(
        local_points_to_world_points_old(camera, _SAMPLE_POINTS),
        jaxcam.local_points_to_world_points(camera, _SAMPLE_POINTS),
    )

  def test_depth_to_ray_depth_regression(self):
    camera = jaxcam.Camera.create(_SAMPLE_ROTATION)

    # Since there are no proper regression tests, the old code is copied here.
    def depth_to_ray_depth_old(
        camera: jaxcam.Camera,
        ray: jnp.ndarray,
        depth: jnp.ndarray,
    ) -> jnp.ndarray:
      cosa = math.matmul(ray, camera.optical_axis)
      return depth / cosa

    rays = jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    depths = jnp.array([1.0, 1.0])
    np.testing.assert_allclose(
        jaxcam.depth_to_ray_depth(camera, rays, depths),
        depth_to_ray_depth_old(camera, rays, depths),
    )

  def test_ray_depth_to_depth_regression(self):
    camera = jaxcam.Camera.create(_SAMPLE_ROTATION)

    # Since there are no proper regression tests, the old code is copied here.
    def ray_depth_to_depth_old(
        camera: jaxcam.Camera,
        ray: jnp.ndarray,
        ray_depth: jnp.ndarray,
    ) -> jnp.ndarray:
      cosa = math.matmul(ray, camera.optical_axis)
      return ray_depth * cosa

    rays = jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    depths = jnp.array([1.0, 1.0])
    np.testing.assert_allclose(
        jaxcam.ray_depth_to_depth(camera, rays, depths),
        ray_depth_to_depth_old(camera, rays, depths),
    )

  @parameterized.product(normalize_rays=[True, False])
  def test_pixels_to_rays_regression(self, normalize_rays: bool):
    camera = jaxcam.Camera.create(
        orientation=_SAMPLE_ROTATION,
    )
    pixels = jnp.array([
        [0.1, 0.1],
        [0.9, 0.9],
        [1.8, 1.8],
    ])

    # Since there are no proper regression tests, the old code is copied here.
    def pixels_to_rays_old(
        camera: jaxcam.Camera,
        pixels: jnp.ndarray,
        normalize: bool,
    ) -> jnp.ndarray:
      if pixels.shape[-1] != 2:
        raise ValueError("The last dimension of pixels must be 2.")

      batch_shape = pixels.shape[:-1]
      pixels = jnp.reshape(pixels, (-1, 2))

      local_rays_dir = jaxcam.pixels_to_local_rays(
          camera,
          pixels,
          normalize=normalize,
      )
      rays_dir = math.matmul(
          jnp.swapaxes(camera.orientation, -1, -2),
          local_rays_dir[..., jnp.newaxis],
      )
      rays_dir = jnp.squeeze(rays_dir, axis=-1)
      rays_dir = rays_dir.reshape((*batch_shape, 3))
      return rays_dir

    np.testing.assert_allclose(
        jaxcam.pixels_to_rays(camera, pixels, normalize_rays),
        pixels_to_rays_old(camera, pixels, normalize_rays),
    )

  @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
  def test_batched_translation(self, batch_shapes):
    batch_shape, *_ = batch_shapes
    camera = jaxcam.Camera.create(
        orientation=jnp.broadcast_to(jnp.eye(3), (*batch_shape, 3, 3)),
    )
    assert camera.translation.shape == (*batch_shape, 3)

  @absltest.skip("TODO: Implement improved batching behavior.")
  @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
  def test_batched_world_points_to_local_points(self, batch_shapes):
    batch_shape_a, batch_shape_b, *_ = batch_shapes
    camera = jaxcam.Camera.create(
        orientation=jnp.broadcast_to(jnp.eye(3), (*batch_shape_a, 3, 3)),
    )
    points = jnp.zeros((*batch_shape_b, 3))
    assert (
        jaxcam.world_points_to_local_points(camera, points).shape
        == (*jnp.broadcast_shapes(batch_shape_a, batch_shape_b), 3)
    )

  @absltest.skip("TODO: Implement improved batching behavior.")
  @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
  def test_batched_local_points_to_world_points(self, batch_shapes):
    batch_shape_a, batch_shape_b, *_ = batch_shapes
    camera = jaxcam.Camera.create(
        orientation=jnp.broadcast_to(jnp.eye(3), (*batch_shape_a, 3, 3)),
    )
    local_points = jnp.zeros((*batch_shape_b, 3))
    assert (
        jaxcam.local_points_to_world_points(camera, local_points).shape
        == (*jnp.broadcast_shapes(batch_shape_a, batch_shape_b), 3)
    )

  @absltest.skip("TODO: Implement improved batching behavior.")
  @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
  def test_batched_depth_to_ray_depth(self, batch_shapes):
    batch_shape_a, batch_shape_b, batch_shape_c = batch_shapes
    camera = jaxcam.Camera.create(
        orientation=jnp.broadcast_to(jnp.eye(3), (*batch_shape_a, 3, 3)),
    )
    rays = jnp.ones((*batch_shape_b, 3))
    depth = jnp.ones(batch_shape_c)
    assert (
        jaxcam.depth_to_ray_depth(camera, rays, depth).shape ==
        jnp.broadcast_shapes(*batch_shapes)
    )

  @absltest.skip("TODO: Implement improved batching behavior.")
  @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
  def test_batched_ray_depth_to_depth(self, batch_shapes):
    batch_shape_a, batch_shape_b, batch_shape_c = batch_shapes
    camera = jaxcam.Camera.create(
        orientation=jnp.broadcast_to(jnp.eye(3), (*batch_shape_a, 3, 3)),
    )
    rays = jnp.ones((*batch_shape_b, 3))
    depth = jnp.ones(batch_shape_c)
    assert (
        jaxcam.ray_depth_to_depth(camera, rays, depth).shape ==
        jnp.broadcast_shapes(*batch_shapes)
    )

  @absltest.skip("TODO: Implement improved batching behavior.")
  @parameterized.product(
      normalize_rays=[True, False],
      batch_shape=_SHAPES_THAT_BROADCAST,
  )
  def test_batched_pixels_to_rays(self, normalize_rays, batch_shape):
    batch_shape_a, batch_shape_b, *_ = batch_shape
    camera = jaxcam.Camera.create(
        orientation=jnp.broadcast_to(_SAMPLE_ROTATION, (*batch_shape_a, 3, 3)),
    )
    pixels = jnp.zeros((*batch_shape_b, 2))
    assert (
        jaxcam.pixels_to_rays(camera, pixels, normalize_rays).shape
        == (*jnp.broadcast_shapes(batch_shape_a, batch_shape_b), 3)
    )

  @absltest.skip("TODO: Implement improved batching behavior.")
  @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
  def test_batched_camera_to_world_matrix(self, batch_shapes):
    batch_shape, *_ = batch_shapes
    cameras = jaxcam.Camera.create(
        orientation=jnp.broadcast_to(jnp.eye(3), (*batch_shape, 3, 3)),
    )
    assert cameras.camera_to_world_matrix.shape == (*batch_shape, 4, 4)


if __name__ == "__main__":
  absltest.main()
