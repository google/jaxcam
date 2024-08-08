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

"""Unit tests for ray utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import jaxcam
from jaxcam._src import rays as jax_rays
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


class RaysTest(parameterized.TestCase):

  @parameterized.parameters(range(20))
  def test_camera_to_ray_to_camera(self, seed):
    rng = jax.random.PRNGKey(seed)
    camera = random_camera(rng)
    directions, origins = jax_rays.get_rays_from_camera(camera)
    camera_recovered = jax_rays.get_camera_from_rays(directions, origins)
    np.testing.assert_allclose(
        camera_recovered.principal_point_x,
        camera.principal_point_x,
        rtol=2e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        camera_recovered.principal_point_y,
        camera.principal_point_y,
        rtol=2e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        camera_recovered.focal_length,
        camera.focal_length,
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        camera_recovered.orientation, camera.orientation, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        camera_recovered.position, camera.position, rtol=1e-3, atol=1e-3
    )

  @parameterized.parameters(range(20))
  def test_camera_to_ray_to_camera_with_noise(self, seed):
    rng = jax.random.PRNGKey(seed)
    sigma = 1e-3
    ransac_paramters = {"num_iterations": 10}

    camera = random_camera(rng)
    rng_direction, rng_origin = jax.random.split(rng, 2)
    directions, origins = jax_rays.get_rays_from_camera(camera)
    noise_direction = (
        jax.random.normal(rng_direction, shape=directions.shape) * sigma
    )
    noise_origin = jax.random.normal(rng_origin, shape=origins.shape) * sigma
    camera_recovered = jax_rays.get_camera_from_rays(
        directions + noise_direction,
        origins + noise_origin,
        use_ransac=True,
        ransac_parameters=ransac_paramters,
    )
    np.testing.assert_allclose(
        camera_recovered.principal_point_x,
        camera.principal_point_x,
        rtol=1e-2,
        atol=10,
    )
    np.testing.assert_allclose(
        camera_recovered.principal_point_y,
        camera.principal_point_y,
        rtol=1e-2,
        atol=10,
    )
    np.testing.assert_allclose(
        camera_recovered.focal_length, camera.focal_length, rtol=1e-2, atol=3
    )
    np.testing.assert_allclose(
        camera_recovered.orientation, camera.orientation, rtol=1e-1, atol=2e-2
    )
    np.testing.assert_allclose(
        camera_recovered.position, camera.position, rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
  absltest.main()
