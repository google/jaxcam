# jaxcam

Jaxcam is a camera library for JAX. Jaxcam is designed to accelerate research by
abstracts cameras and common operations through a simple API.


[![Unittests](https://github.com/google/jaxcam/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google/jaxcam/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/jaxcam.svg)](https://badge.fury.io/py/jaxcam)

*This is not an officially supported Google product.*


## Usage Examples

Basic imports

```python
from jax import numpy as jnp
from jax import random
import jaxcam
```

Creating a camera:

```python
>> camera = jaxcam.Camera.create(
     orientation=jnp.eye(3),
     position=jnp.array([0.0, 0.0, 5.0]),
     image_size=jnp.array([512, 512]),
     focal_length=jnp.array(512)
   )
>> camera
Camera(orientation=Array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]], dtype=float32), position=Array([0., 0., 5.], dtype=float32), focal_length=Array(512., dtype=float32), principal_point=Array([256., 256.], dtype=float32), image_size=Array([512., 512.], dtype=float32), skew=Array(0., dtype=float32), pixel_aspect_ratio=Array(1., dtype=float32), radial_distortion=None, tangential_distortion=None, projection_type=<ProjectionType.PERSPECTIVE: 'perspective'>, use_inverted_distortion=False)
```

Projecting 3D world points to camera pixels:

```python
>> points = random.normal(random.PRNGKey(0), (5, 3))
>> pixels = jaxcam.project(camera, points)
>> pixels
Array([[246.76802 ,  48.151352],
       [157.49011 , 226.61935 ],
       [198.61266 , 347.66937 ],
       [311.53983 , 178.945   ],
       [281.20334 , 342.3573  ]], dtype=float32)

```
