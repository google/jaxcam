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


if __name__ == "__main__":
  absltest.main()
