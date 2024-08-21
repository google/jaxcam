# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google/jaxcam/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

## [0.3.0] - 2024-08-20

* Added Rays dataclass with Plucker ray support.
* Added utilities for transforming cameras.

[Unreleased]: https://github.com/google/jaxcam/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/google/jaxcam/releases/tag/v0.3.0

## [0.2.0] - 2024-08-08

* Added utilities to convert Cameras to rays and rays to Cameras.

[Unreleased]: https://github.com/google/jaxcam/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/google/jaxcam/releases/tag/v0.2.0

## [0.1.1] - 2022-01-01

* Initial release

[Unreleased]: https://github.com/google/jaxcam/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/google/jaxcam/releases/tag/v0.1.1
