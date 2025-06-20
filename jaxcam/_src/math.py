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

"""Math utilities."""

from typing import Union

import jax
from jax import numpy as jnp
import numpy as onp
from numpy import typing as npt


def einsum(subscripts: str, /, *operands: npt.ArrayLike) -> npt.ArrayLike:
  """jnp.einsum uses bfloat16 by default on TPU, this prevents that.

  References:
    https://github.com/jax-ml/jax#current-gotchas
    https://github.com/jax-ml/jax/issues/2161
    https://github.com/jax-ml/jax/issues/7010

  Args:
    subscripts: The einsum subscripts.
    *operands: The operands for the einsum.

  Returns:
    The result of xnp.einsum(subscripts, *operands, precision=HIGHEST)
  """
  xnp = operands[0].__array_namespace__()
  if xnp is jnp:
    return jnp.einsum(
        subscripts, *operands, precision=jax.lax.Precision.HIGHEST
    )
  elif xnp is onp:
    return onp.einsum(subscripts, *operands)
  else:
    raise ValueError(f'Unsupported numpy-like module: {xnp}')


def matmul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
  """Matrix-matrix product that prevents uses bfloat16 by default on TPU."""
  xnp = a.__array_namespace__()
  if xnp is jnp:
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
  elif xnp is onp:
    return onp.matmul(a, b)
  else:
    raise ValueError(f'Unsupported numpy-like module: {xnp}')


def matvecmul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
  """Matrix-vector product that prevents using bfloat16 on TPU."""
  return einsum('...ij,...j->...i', a, b)


def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
  """Vector-vector dot product that prevents using bfloat16 on TPU."""
  return einsum('...i,...i->...', a, b)


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
  xnp = vector.__array_namespace__()
  vector = xnp.reshape(vector, (3))
  return xnp.array([
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
