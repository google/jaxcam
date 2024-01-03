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

"""Functions for reading and writing Cameras."""

import json
import pathlib
from typing import Union

from etils import epath
import jaxcam
import numpy as np

PathType = Union[str, pathlib.PurePath]


def from_nerfies_json_file(path: PathType) -> jaxcam.Camera:
  """Load a camera from a JSON file in the Nerfies format.

  This format directly serializes the fields of the camera dataclass, and
  additionally an `image_size` field containing `(image_size_x, image_size_y)`.

  This format is used by the Nerfies datasets, which is available publicly at
  https://github.com/google/nerfies#camera

  Args:
    path: the path of the file to read.

  Returns:
    The loaded camera instance.
  """
  path = epath.Path(path)
  with path.open('r') as fp:
    camera_dict = json.load(fp)
  camera_dict = {k: np.asarray(v) for k, v in camera_dict.items()}
  return jaxcam.Camera.create(**camera_dict)


def to_nerfies_json_file(path: PathType, camera: jaxcam.Camera) -> None:
  """Saves a camera to a JSON file in the Nerfies format.

  Please see :obj:`from_nerfies_json_file` for details.

  Args:
    path: the path to save to.
    camera: the camera to save.
  """
  path = epath.Path(path)
  camera_dict = {
      k: v.tolist() if hasattr(v, 'tolist') else v
      for k, v in camera.pytree_dict().items()
  }
  with path.open('w') as fp:
    json.dump(camera_dict, fp, indent=2)
