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

"""The public interface camera I/O."""

# pylint: disable=g-multiple-import,g-importing-member,useless-import-alias,unused-import
# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570
from jaxcam._src.io import (
    from_nerfies_json_file as from_nerfies_json_file,
    to_nerfies_json_file as to_nerfies_json_file,
)
