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
from jax import random
import jax.numpy as jnp
import jaxcam
from jaxcam._src import camera_test_util
import numpy as onp


class CameraCrossPlatformTest(parameterized.TestCase):

  def _replace_backend_helper(self, anp, bnp):
    """Replace the backend from `anp` to `bnp`, and check that it worked."""
    camera_anp = camera_test_util.random_camera(random.PRNGKey(0), anp)
    camera_bnp = jaxcam.replace_backend(camera_anp, bnp)

    self.assertEqual(camera_bnp.xnp, bnp)

    for field in set(vars(camera_anp).keys()).difference({'xnp'}):
      a = getattr(camera_anp, field)
      b = getattr(camera_bnp, field)

      if isinstance(a, anp.ndarray):
        self.assertIsInstance(b, bnp.ndarray)
        onp.testing.assert_allclose(a, b, atol=1e-15)
      else:
        self.assertEqual(a, b)

  def test_jax_to_numpy(self):
    self._replace_backend_helper(jnp, onp)

  def test_numpy_to_jax(self):
    self._replace_backend_helper(onp, jnp)


if __name__ == '__main__':
  absltest.main()
