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

from absl.testing import absltest
import jax.numpy as jnp
from jaxcam._src import camera_test_util


CameraJnpTest = camera_test_util.create_camera_test_class(jnp, 'Jnp')

if __name__ == '__main__':
  absltest.main()
