# Jensen source-fitted candidate v3 — attribution and license

This is a research simulation and an explicitly selectable, unaccepted likeness
candidate. Its surface was newly fitted to Jensen Huang references on a generic
anatomical foundation. It is an articulated three-dimensional model with facial
morphs and skinned hands. No ready-made celebrity mesh was used.

## Adapted photographs and distributed asset

The distributed fitted model and its adapted facial texture are provided under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

Face texture adapted from these photographs by **Anders Eidesvik
(Anderseidesvik)**, originally published under CC BY-SA 4.0:

- [Jensen Huang, Stanford, 2026-04-30, photograph 008](https://commons.wikimedia.org/wiki/File:Jensen_huang_stanford_2026-04-30_008.jpg), source page revision 1298252096.
- [Jensen Huang, Stanford, 2026-04-30, photograph 010](https://commons.wikimedia.org/wiki/File:Jensen_huang_stanford_2026-04-30_010.jpg), source page revision 1300280320.

Changes by the Who Is Your Bias project: source landmark alignment, multi-view
surface deformation, image warping and blending into the animated skin UV map,
body color matching, eye and glasses exclusions, and neutral-mouth source
selection. Glasses, dark jacket appearance, iris color, hair appearance and
eyebrow adjustments are authored choices. Photographic lighting can remain in
the derived texture; this is not recovered intrinsic albedo or a calibrated scan.

Retain the attribution, indicate your changes, and distribute adaptations under
the same license. No endorsement by the photographer, Jensen Huang, Stanford,
or NVIDIA is implied.

## Generic foundation

The pre-existing `research-human-v2.glb` anatomical foundation, MakeHuman/MPFB
graphical ingredients and extra expression targets were CC0-1.0 inputs. Their
source archives, revisions and SHA-256 values are recorded in
`tools/avatar/inputs.json`; the retained license is `tools/avatar/LICENSE-CC0.txt`.
The MPFB build tool has its own GPL-3.0 software license, separate from its
CC0 graphical assets. The photograph-derived asset above is not labeled CC0.

## Geometry references and motion

NVIDIA executive portraits, Stanford photographs and Goldman Sachs interview
frames contributed source-bound facial landmark measurements. Only the two
licensed Stanford photographs contribute source image pixels to the texture.
Motion observations and their attribution are separate, explicitly preview-only
clip files; animation does not establish that a new utterance is an observed
statement or gesture by the source person.

The reproducible recipes, input hashes, fitted versus authored distinctions,
editable-master receipts, actual validation and remaining quality gates are in
`reports/jensen-avatar/README.md` and its linked result files.
