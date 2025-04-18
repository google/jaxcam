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

"""Math utilities."""

from typing import Union

import jax
from jax import numpy as jnp
from numpy import typing as npt


def einsum(*args, **kwargs) -> npt.ArrayLike:
  """jnp.einsum uses bfloat16 by default on TPU, this prevents that."""
  return jnp.einsum(*args, **kwargs, precision=jax.lax.Precision.HIGHEST)


def matmul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
  """jnp.matmul uses bfloat16 by default on TPU, this prevents that.

  Use matvecmul for matrix-vector products and dot for vector-vector dot
  products.

  References:
    https://github.com/jax-ml/jax#current-gotchas
    https://github.com/jax-ml/jax/issues/2161
    https://github.com/jax-ml/jax/issues/7010

  Args:
    a: the left matrix of a matmul.
    b: the right matrix of a matmul.

  Returns:
    The result of the matmul.
  """
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def matvecmul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
  """Matrix-vector product that prevents using bfloat16 on TPU.

  References:
    https://github.com/jax-ml/jax#current-gotchas
    https://github.com/jax-ml/jax/issues/2161
    https://github.com/jax-ml/jax/issues/7010

  Args:
    a: the left matrix of a matrix-vector product.
    b: the right vector of a matrix-vector product.

  Returns:
    The result of the matrix-vector product.
  """
  return einsum("...ij,...j->...i", a, b)


def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
  """Vector-vector dot product that prevents using bfloat16 on TPU.

  References:
    https://github.com/jax-ml/jax#current-gotchas
    https://github.com/jax-ml/jax/issues/2161
    https://github.com/jax-ml/jax/issues/7010

  Args:
    a: the left vector of a dot product.
    b: the right vector of a dot product.

  Returns:
    The result of the dot product.
  """
  return einsum("...i,...i->...", a, b)


def skew(vector: npt.ArrayLike) -> npt.ArrayLike:
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
    point: npt.ArrayLike,
    scale: Union[float, npt.ArrayLike],
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike,
) -> npt.ArrayLike:
  """Transforms the given point x as `scale * rotation @ x + translation`."""
  return scale * matmul(rotation, point) + translation
