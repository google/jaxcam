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

# pylint: disable=g-multiple-import,g-importing-member,useless-import-alias
"""The main package namespace."""

# A new PyPI release will be pushed everytime `__version__` is increased
# When changing this, also update the CHANGELOG.md
__version__ = '0.1.1'

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570
from jaxcam._src.camera import (
    Camera as Camera,
    concatenate as concatenate,
    create as create,
    crop as crop,
    depth_to_ray_depth as depth_to_ray_depth,
    essential_matrix as essential_matrix,
    fundamental_matrix as fundamental_matrix,
    get_pixel_centers as get_pixel_centers,
    invert_radial_distortion_coefficients as invert_radial_distortion_coefficients,
    local_points_to_world_points as local_points_to_world_points,
    look_at as look_at,
    pixels_to_local_rays as pixels_to_local_rays,
    pixels_to_points as pixels_to_points,
    pixels_to_rays as pixels_to_rays,
    project as project,
    ray_depth_to_depth as ray_depth_to_depth,
    relative_transform as relative_transform,
    scale as scale,
    transform as transform,
    update_intrinsic_matrix as update_intrinsic_matrix,
    update_translation as update_translation,
    update_world_to_camera_matrix as update_world_to_camera_matrix,
    world_points_to_local_points as world_points_to_local_points,
    world_to_camera_matrix as world_to_camera_matrix,
)
