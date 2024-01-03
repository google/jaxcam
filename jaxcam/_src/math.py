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

"""Math utilities."""

import jax
from jax import numpy as jnp


def matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  """jnp.matmul uses bfloat16 by default on TPU, this prevents that.

  References:
    https://github.com/google/jax#current-gotchas
    https://github.com/google/jax/issues/2161
    https://github.com/google/jax/issues/7010

  Args:
    a: the left matrix of a matmul.
    b: the right matrix of a matmul.

  Returns:
    The result of the matmul.
  """
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def einsum(*args, **kwargs) -> jnp.ndarray:
  """jnp.einsum uses bfloat16 by default on TPU, this prevents that."""
  return jnp.einsum(*args, **kwargs, precision=jax.lax.Precision.HIGHEST)


def skew(vector: jnp.ndarray) -> jnp.ndarray:
  """Builds a skew matrix ("cross product matrix") for a vector.

  References:
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix.

  Args:
    vector: (3,) A 3-vector.

  Returns:
    A (3, 3) skew-symmetric matrix such that W @ v == w x v where w is the input
    vector, W is the corresponding skew-symmatrix matrix, and v is another
    vector.
  """
  vector = jnp.reshape(vector, (3))
  return jnp.array([
      [0.0, -vector[2], vector[1]],
      [vector[2], 0.0, -vector[0]],
      [-vector[1], vector[0], 0.0],
  ])


def transform_point(
    point: jnp.ndarray,
    scale: jnp.ndarray,
    rotation: jnp.ndarray,
    translation: jnp.ndarray,
) -> jnp.ndarray:
  """Transforms the given point x as `scale * rotation @ x + translation`."""
  return scale * matmul(rotation, point) + translation
