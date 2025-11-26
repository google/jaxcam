# Copyright 2025 The jaxcam Authors.
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

import pickle

from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import jaxcam
from jaxcam._src import math
import numpy as onp


_IMAGE_WIDTH = 512
_IMAGE_HEIGHT = 512
_FOCAL_LENGTH = 512
_SAMPLE_ROTATION = onp.array([
    [0.22629565, -0.18300793, 0.95671225],
    [0.95671225, 0.22629565, -0.18300793],
    [-0.18300793, 0.95671225, 0.22629565],
])
_SAMPLE_POINTS = onp.array([
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
    key,
    xnp,
    radius: int = 1,
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
):
  """Creates a random camera for testing."""
  position = random.normal(key, (3,))
  position = radius * position / onp.linalg.norm(position)
  camera = jaxcam.Camera.create(
      xnp=xnp,
      focal_length=xnp.array(_FOCAL_LENGTH),
      image_size=xnp.array((_IMAGE_WIDTH, _IMAGE_HEIGHT)),
  )
  camera = jaxcam.look_at(
      camera,
      eye=xnp.asarray(position),
      center=xnp.array(look_at),
      world_up=xnp.array(up),
  )
  return camera


def create_camera_test_class(xnp, xnp_name: str):
  """Creates a CameraTest class parameterized by the numpy implementation."""

  class CameraTestBase(parameterized.TestCase):
    """Base class for camera tests."""

    def setUp(self):
      super().setUp()
      self.rng = random.PRNGKey(0)

    def test_world_to_camera_is_inverse_of_camera_to_world_matrix(self):
      """Tests that wold_to_camera is the inverse of camera_to_world."""
      camera = random_camera(self.rng, xnp)

      onp.testing.assert_allclose(
          camera.world_to_camera_matrix,
          onp.linalg.inv(camera.camera_to_world_matrix),
          atol=1e-6,
      )
      onp.testing.assert_allclose(
          camera.camera_to_world_matrix,
          onp.linalg.inv(camera.world_to_camera_matrix),
          atol=1e-6,
      )
      onp.testing.assert_allclose(
          camera.world_to_camera_matrix @ camera.camera_to_world_matrix,
          xnp.eye(4),
          atol=1e-6,
      )

    @parameterized.product(
        scale=[0.1, 1.0, 2.0],
        rotation=[
            onp.eye(3),
            onp.array([
                [1.0, 0.0, 0.0],
                [0.0, -0.41614687, -0.9092974],
                [0.0, 0.9092974, -0.41614687],
            ]),
            onp.array([
                [0.7071067, 0.0, 0.7071068],
                [0.0, 1.0, 0.0],
                [-0.7071068, 0.0, 0.7071067],
            ]),
            onp.array([
                [0.7071067, -0.7071068, 0.0],
                [0.7071068, 0.7071067, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            onp.array([
                [0.22629565, -0.18300793, 0.95671225],
                [0.95671225, 0.22629565, -0.18300793],
                [-0.18300793, 0.95671225, 0.22629565],
            ]),
        ],
        translation=[
            onp.zeros(3),
            onp.ones(3),
            onp.array([0.1, 0.0, -0.1]),
        ],
    )
    def test_transform_projection_identical(self, scale, rotation, translation):
      """Tests that transform->project is the same as project->transform."""
      camera = random_camera(self.rng, xnp)
      points = random.uniform(self.rng, (100, 3), minval=-0.2, maxval=0.2)

      points = xnp.asarray(points)
      rotation = xnp.asarray(rotation)
      translation = xnp.asarray(translation)

      projected_points = jaxcam.project(camera, points)
      transformed_camera = jaxcam.transform(
          camera, scale, rotation, translation
      )

      vectorized_transform = xnp.vectorize(
          math.transform_point, signature='(d),(),(d,d),(d)->(d)'
      )
      transformed_points = vectorized_transform(
          points, scale, rotation, translation
      )
      transformed_projected_points = jaxcam.project(
          transformed_camera, transformed_points
      )

      onp.testing.assert_allclose(
          projected_points, transformed_projected_points, atol=1e-3
      )

    def test_world_points_to_local_points_regression(self):
      """Tests that world_points_to_local_points matches the old impl."""
      camera = jaxcam.Camera.create(
          xnp=xnp,
          orientation=xnp.asarray(_SAMPLE_ROTATION),
          position=xnp.array([0.5, 1.5, 2.5]),
      )
      sample_points = xnp.asarray(_SAMPLE_POINTS)

      def world_points_to_local_points_old(
          camera_pos: onp.ndarray,
          camera_orient: onp.ndarray,
          world_points: onp.ndarray,
      ) -> onp.ndarray:
        world_points_flat = world_points.reshape((-1, 3))
        translated_points = world_points_flat - camera_pos
        local_points = (math.matmul(camera_orient, translated_points.T)).T
        return local_points.reshape(world_points.shape)

      expected = world_points_to_local_points_old(
          onp.array([0.5, 1.5, 2.5]), _SAMPLE_ROTATION, _SAMPLE_POINTS
      )
      actual = jaxcam.world_points_to_local_points(camera, sample_points)

      onp.testing.assert_allclose(expected, actual, atol=1e-6)

    def test_local_points_to_world_points_regression(self):
      """Tests that local_points_to_world_points matches the old impl."""
      camera = jaxcam.Camera.create(
          xnp=xnp,
          orientation=xnp.asarray(_SAMPLE_ROTATION),
          position=xnp.array([0.5, 1.5, 2.5]),
      )
      sample_points = xnp.asarray(_SAMPLE_POINTS)

      def local_points_to_world_points_old(
          camera_pos: onp.ndarray,
          camera_orient: onp.ndarray,
          local_points: onp.ndarray,
      ) -> onp.ndarray:
        local_points_flat = local_points.reshape((-1, 3))
        rotated_points = math.matmul(camera_orient.T, local_points_flat.T).T
        world_points = rotated_points + camera_pos
        return world_points.reshape(local_points.shape)

      expected = local_points_to_world_points_old(
          onp.array([0.5, 1.5, 2.5]), _SAMPLE_ROTATION, _SAMPLE_POINTS
      )
      actual = jaxcam.local_points_to_world_points(camera, sample_points)

      onp.testing.assert_allclose(expected, actual, atol=1e-6)

    def test_depth_to_ray_depth_regression(self):
      """Tests that depth_to_ray_depth matches the old implementation."""
      camera = jaxcam.Camera.create(_SAMPLE_ROTATION)

      # Since there are no proper regression tests, the old code is copied here.
      def depth_to_ray_depth_old(
          camera: jaxcam.Camera,
          ray: onp.ndarray,
          depth: onp.ndarray,
      ) -> onp.ndarray:
        cosa = math.matmul(ray, camera.optical_axis)
        return depth / cosa

      rays = onp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
      depths = onp.array([1.0, 1.0])
      onp.testing.assert_allclose(
          jaxcam.depth_to_ray_depth(camera, rays, depths),
          depth_to_ray_depth_old(camera, rays, depths),
      )

    def test_ray_depth_to_depth_regression(self):
      """Tests that ray_depth_to_depth matches the old implementation."""
      camera = jaxcam.Camera.create(_SAMPLE_ROTATION)

      # Since there are no proper regression tests, the old code is copied here.
      def ray_depth_to_depth_old(
          camera: jaxcam.Camera,
          ray: onp.ndarray,
          ray_depth: onp.ndarray,
      ) -> onp.ndarray:
        cosa = math.matmul(ray, camera.optical_axis)
        return ray_depth * cosa

      rays = onp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
      depths = onp.array([1.0, 1.0])
      onp.testing.assert_allclose(
          jaxcam.ray_depth_to_depth(camera, rays, depths),
          ray_depth_to_depth_old(camera, rays, depths),
      )

    @parameterized.product(normalize_rays=[True, False])
    def test_pixels_to_rays_regression(self, normalize_rays: bool):
      """Tests that pixels_to_rays matches the old implementation."""
      camera = jaxcam.Camera.create(
          xnp=xnp,
          orientation=xnp.asarray(_SAMPLE_ROTATION),
      )
      pixels = onp.array([
          [0.1, 0.1],
          [0.9, 0.9],
          [1.8, 1.8],
      ])

      # Since there are no proper regression tests, the old code is copied here.
      def pixels_to_rays_old(
          camera: jaxcam.Camera,
          pixels: onp.ndarray,
          normalize: bool,
      ) -> onp.ndarray:
        if pixels.shape[-1] != 2:
          raise ValueError('The last dimension of pixels must be 2.')

        batch_shape = pixels.shape[:-1]
        pixels_flat = onp.reshape(pixels, (-1, 2))

        local_rays_dir = jaxcam.pixels_to_local_rays(
            camera,
            pixels_flat,
            normalize=normalize,
        )
        rays_dir = math.matmul(
            onp.swapaxes(camera.orientation, -1, -2),
            local_rays_dir[..., onp.newaxis],
        )
        rays_dir = onp.squeeze(rays_dir, axis=-1)
        rays_dir = rays_dir.reshape((*batch_shape, 3))
        return rays_dir

      onp.testing.assert_allclose(
          jaxcam.pixels_to_rays(camera, pixels, normalize_rays),
          pixels_to_rays_old(camera, pixels, normalize_rays),
      )

    @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
    def test_batched_translation(self, batch_shapes):
      """Tests that batched cameras have the correct translation shape."""
      batch_shape, *_ = batch_shapes
      orientation_xnp = xnp.broadcast_to(xnp.eye(3), (*batch_shape, 3, 3))
      camera = jaxcam.Camera.create(xnp=xnp, orientation=orientation_xnp)
      expected_shape = onp.broadcast_shapes(batch_shape, ())
      self.assertEqual(camera.translation.shape, (*expected_shape, 3))

    @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
    def test_batched_world_points_to_local_points(self, batch_shapes):
      """Tests that batched world_points_to_local_points has the right shape."""
      batch_shape_a, batch_shape_b, *_ = batch_shapes
      orientation_xnp = xnp.broadcast_to(xnp.eye(3), (*batch_shape_a, 3, 3))
      camera = jaxcam.Camera.create(xnp=xnp, orientation=orientation_xnp)
      points = xnp.zeros((*batch_shape_b, 3))
      expected_shape = onp.broadcast_shapes(batch_shape_a, batch_shape_b)
      self.assertEqual(
          jaxcam.world_points_to_local_points(camera, points).shape,
          (*expected_shape, 3),
      )

    @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
    def test_batched_local_points_to_world_points(self, batch_shapes):
      """Tests that batched local_points_to_world_points has the right shape."""
      batch_shape_a, batch_shape_b, *_ = batch_shapes
      orientation_xnp = xnp.broadcast_to(xnp.eye(3), (*batch_shape_a, 3, 3))
      camera = jaxcam.Camera.create(xnp=xnp, orientation=orientation_xnp)
      local_points = xnp.zeros((*batch_shape_b, 3))
      expected_shape = onp.broadcast_shapes(batch_shape_a, batch_shape_b)
      self.assertEqual(
          jaxcam.local_points_to_world_points(camera, local_points).shape,
          (*expected_shape, 3),
      )

    @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
    def test_batched_depth_to_ray_depth(self, batch_shapes):
      """Tests that batched depth_to_ray_depth has the correct shape."""
      batch_shape_a, batch_shape_b, batch_shape_c = batch_shapes
      orientation_xnp = xnp.broadcast_to(xnp.eye(3), (*batch_shape_a, 3, 3))
      camera = jaxcam.Camera.create(xnp=xnp, orientation=orientation_xnp)
      rays = xnp.ones((*batch_shape_b, 3))
      depth = xnp.ones(batch_shape_c)
      expected_shape = onp.broadcast_shapes(*batch_shapes)
      self.assertEqual(
          jaxcam.depth_to_ray_depth(camera, rays, depth).shape, expected_shape
      )

    @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
    def test_batched_ray_depth_to_depth(self, batch_shapes):
      """Tests that batched ray_depth_to_depth has the correct shape."""
      batch_shape_a, batch_shape_b, batch_shape_c = batch_shapes
      orientation_xnp = xnp.broadcast_to(xnp.eye(3), (*batch_shape_a, 3, 3))
      camera = jaxcam.Camera.create(xnp=xnp, orientation=orientation_xnp)
      rays = xnp.ones((*batch_shape_b, 3))
      depth = xnp.ones(batch_shape_c)
      expected_shape = onp.broadcast_shapes(*batch_shapes)
      self.assertEqual(
          jaxcam.ray_depth_to_depth(camera, rays, depth).shape, expected_shape
      )

    @parameterized.product(
        normalize_rays=[True, False],
        batch_shape=_SHAPES_THAT_BROADCAST,
    )
    def test_batched_pixels_to_rays(self, normalize_rays, batch_shape):
      """Tests that batched pixels_to_rays has the correct shape."""
      batch_shape_a, batch_shape_b, *_ = batch_shape
      orientation_xnp = xnp.broadcast_to(
          xnp.asarray(_SAMPLE_ROTATION), (*batch_shape_a, 3, 3)
      )
      camera = jaxcam.Camera.create(xnp=xnp, orientation=orientation_xnp)
      pixels = xnp.zeros((*batch_shape_b, 2))
      expected_shape = onp.broadcast_shapes(batch_shape_a, batch_shape_b)
      self.assertEqual(
          jaxcam.pixels_to_rays(camera, pixels, normalize_rays).shape,
          (*expected_shape, 3),
      )

    @parameterized.product(batch_shapes=_SHAPES_THAT_BROADCAST)
    def test_batched_camera_to_world_matrix(self, batch_shapes):
      """Tests that batched camera_to_world_matrix has the correct shape."""
      batch_shape, *_ = batch_shapes
      orientation_xnp = xnp.broadcast_to(xnp.eye(3), (*batch_shape, 3, 3))
      cameras = jaxcam.Camera.create(xnp=xnp, orientation=orientation_xnp)
      expected_shape = onp.broadcast_shapes(batch_shape, ())
      self.assertEqual(
          cameras.camera_to_world_matrix.shape, (*expected_shape, 4, 4)
      )

    @parameterized.product(
        batch_shapes=[
            ((3, 8), (24,)),
            ((3, 8), (24, 1)),
            ((3, 8), (8, 3)),
            ((3, 8), (1, 2, 3, 4)),
            ((24,), (3, 8)),
        ]
    )
    def test_camera_reshape(self, batch_shapes):
      """Tests that camera reshape works correctly."""
      shape1, shape2 = batch_shapes
      orientation1 = xnp.broadcast_to(xnp.eye(3), shape1 + (3, 3))
      camera1 = jaxcam.Camera.create(xnp=xnp, orientation=orientation1)
      camera2 = camera1.reshape(shape2)
      self.assertEqual(camera2.shape, shape2)
      self.assertEqual(camera2.orientation.shape, shape2 + (3, 3))

    def test_pickle_unpickle(self):
      """Tests that pickling and unpickling works correctly."""
      camera = jaxcam.Camera.create(
          xnp=xnp,
          orientation=xnp.asarray(_SAMPLE_ROTATION),
          position=xnp.array([0.5, 1.5, 2.5]),
      )
      pickled_camera = pickle.dumps(camera)
      unpickled_camera = pickle.loads(pickled_camera)
      self.assertEqual(unpickled_camera.xnp, xnp)

    @parameterized.product(
        depth_values=[
            xnp.finfo(xnp.float32).smallest_normal * 0.1,
            -xnp.finfo(xnp.float32).smallest_normal * 0.1,
        ],
        orientation=[
            onp.array([
                [1.0, 0.0, 0.0],
                [0.0, -0.41614687, -0.9092974],
                [0.0, 0.9092974, -0.41614687],
            ]),
            onp.array([
                [0.7071067, 0.0, 0.7071068],
                [0.0, 1.0, 0.0],
                [-0.7071068, 0.0, 0.7071067],
            ]),
            onp.array([
                [0.7071067, -0.7071068, 0.0],
                [0.7071068, 0.7071067, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            onp.array(
                [
                    [0.22629565, -0.18300793, 0.95671225],
                    [0.95671225, 0.22629565, -0.18300793],
                    [-0.18300793, 0.95671225, 0.22629565],
                ],
            ),
        ],
        position=[
            onp.zeros(3),
            onp.ones(3),
            onp.array([0.1, 0.0, -0.1]),
        ],
        is_fisheye=[
            False,
            True,
        ],
    )
    def test_camera_project_does_not_return_nans(
        self, depth_values, orientation, position, is_fisheye
    ):
      """Tests that the camera project function does not return NaNs."""
      # Enable NaNs silencing.
      with jaxcam.silence_nans(True):
        camera = jaxcam.Camera.create(
            xnp=xnp,
            orientation=xnp.array(orientation),
            position=xnp.array(position),
            is_fisheye=is_fisheye,
        )

        points = xnp.asarray(
            [
                [0.0, 0.0, depth_values],
                [1.0, 0.0, depth_values],
                [0.0, 1.0, depth_values],
                [1.0, 1.0, depth_values],
            ],
            dtype=xnp.float32,
        )

        projections = jaxcam.project(camera, points)
        self.assertFalse(xnp.isnan(projections).any())

        if xnp is jnp:

          def loss(points, camera):
            projections = jaxcam.project(camera, points)
            return jnp.sum(jnp.square(projections))

          grad_func = jax.value_and_grad(
              loss,
              argnums=(0, 1),
          )

          loss_value, grads = grad_func(points, camera)
          self.assertFalse(jnp.isnan(loss_value).any())

          flat_grads = jax.tree.leaves(grads)

          has_nan_grads = any(
              jax.tree.map(
                  lambda x: jnp.any(x).item(),
                  jax.tree.map(jnp.isnan, flat_grads),
              )
          )
          self.assertFalse(has_nan_grads)

    @parameterized.product(
        value=[True, False],
    )
    def test_silence_nans_context_manager(self, value: bool):
      """Tests that the silence_nans context manager works correctly."""
      with jaxcam.silence_nans(value):
        self.assertEqual(jaxcam.get_silence_nans(), value)
      self.assertEqual(jaxcam.get_silence_nans(), False)

    @parameterized.product(
        scale_factor=[0.5, 2.0],
        principal_point_offset=[
            onp.array([0.0, 0.0]),
            onp.array([-16.0, 12.0]),
        ],
        radial_distortion=[
            None,
            onp.array([0.01, 0.001, 0.0001, 0.0]),
            onp.array([0.4, 0.2, 0.1, 0.0]),
        ],
        tangential_distortion=[
            None,
            onp.array([0.001, 0.0001]),
            onp.array([0.04, 0.02]),
        ],
        invert_distortion=[True, False],
    )
    def test_scale_camera(
        self,
        scale_factor,
        principal_point_offset,
        radial_distortion,
        tangential_distortion,
        invert_distortion,
    ):
      """Tests that scale() works with distortion."""
      if (
          radial_distortion is not None or tangential_distortion is not None
      ) and xnp is onp:
        self.skipTest('numpy backend does not support distortion.')

      if radial_distortion is not None:
        radial_distortion = xnp.array(radial_distortion)
      if tangential_distortion is not None:
        tangential_distortion = xnp.array(tangential_distortion)

      width = 512
      height = 256
      camera = jaxcam.Camera.create(
          xnp=xnp,
          focal_length=xnp.array(512.0),
          image_size=xnp.array([width, height]),
          principal_point=xnp.array([width / 2, height / 2])
          + xnp.array(principal_point_offset),
          radial_distortion=radial_distortion,
          tangential_distortion=tangential_distortion,
          invert_distortion=invert_distortion,
      )

      scaled_camera = jaxcam.scale(camera, scale_factor)

      pixels = jaxcam.get_pixel_centers(width, height)
      rays = jaxcam.pixels_to_rays(camera, pixels)
      scaled_pixels = pixels * scale_factor
      scaled_rays = jaxcam.pixels_to_rays(scaled_camera, scaled_pixels)
      onp.testing.assert_allclose(rays, scaled_rays, atol=1e-5)

  CameraTestBase.__name__ = f'CameraTest_{xnp_name}'
  return CameraTestBase
