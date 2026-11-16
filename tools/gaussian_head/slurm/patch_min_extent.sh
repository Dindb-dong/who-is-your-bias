#!/bin/bash
# One-line patch to GaussianAvatars scene/__init__.py: monocular captures give cameras_extent = 0,
# which disables opacity pruning and clone after 60k (see docs/jensen-gaussian-head.md, section 3).
# GA_MIN_EXTENT (env) sets a lower bound; unset/0 keeps upstream behaviour. Idempotent.
set -eo pipefail
GA=${1:?usage: patch_min_extent.sh <GaussianAvatars checkout>}
F=$GA/scene/__init__.py
grep -q GA_MIN_EXTENT "$F" || sed -i 's#^        self.cameras_extent = scene_info.nerf_normalization\["radius"\]$#        self.cameras_extent = max(scene_info.nerf_normalization["radius"], float(os.environ.get("GA_MIN_EXTENT", "0"))); print("cameras_extent", self.cameras_extent)#' "$F"
grep -n "cameras_extent =" "$F"
grep -q GA_MIN_EXTENT "$F" && echo "PATCH OK"
