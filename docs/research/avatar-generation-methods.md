# Avatar generation methods: evidence ledger

Accessed **2026-11-11 UTC / 2026-11-12 KST**. This records upstream documentation,
source and license inspection. These methods were **not run locally** for this
review; no new avatar, reconstruction benchmark or quality winner was produced.

The scope is generating a new likeness from multiple photographs and continuous
video, with a complete upper torso, both arms/hands, facial expressions and speech
animation. The implementation plan lives in [Avatar generation pipeline](../avatar-generation-pipeline.md).

## Representation and integration boundaries

The proposed product artifact is a textured mesh with body/finger skinning and
facial morph targets. glTF supports skins, morph targets and animation, but naming
controls consistently does not create missing deformation data. See the
[glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
and this repository's [avatar runtime contract](../../apps/web/src/avatar/README.md).

| Method | Documented input | Documented output | Role in this plan |
| --- | --- | --- | --- |
| ExAvatar | Monocular person video | Expressive Gaussian avatar with SMPL-X-based deformation | Coherent-video experiment |
| LHM++ | One or multiple pose-free images | Animatable Gaussian representation; selected variants export GS PLY | Multi-photo experiment |
| MPFB / Blender | Authored human parameters and mesh edits | Body mesh, rigs and optional facial shape keys | Production mesh foundation |
| Anny | Shape, pose and facial-action parameters | Differentiable human mesh | Optional fitting component |
| MetaHuman 5.8 | Custom human head/body mesh | MetaHuman topology and rig through the complete workflow | Optional mesh-authoring route |
| Charakuru | Character illustration and staged production inputs | Blender/VRM workflow artifacts | Automation design reference |

The evidence for each row is linked in its section below. These are roles, not a
ranking of identity fidelity. No method has been evaluated on this project's data.

**Gaussian PLY is not an animated GLB.** LHM++ describes its PLY as Gaussian data;
ExAvatar renders Gaussian appearance despite using mesh connectivity. A splat
viewer, one posed export or a rendered video does not establish a portable facial
rig. Gaussian-to-mesh reconstruction, topology cleanup, texturing and re-rigging
would be additional, unverified work; changing the extension does not perform it.
Sources: [LHM++ export documentation](https://github.com/aigc3d/LHM-plusplus#readme),
[ExAvatar representation](https://mks0601.github.io/ExAvatar/).

## Source data determines which experiment is meaningful

Use a coherent shot or capture session for the first reconstruction trial. Keep
separate records for source URLs, timestamps, usage terms, crops, masks, identity
references, reconstruction frames, held-out views and motion-reference clips.
This is a proposed data protocol, not a claim that arbitrary public footage works.

COLMAP's guidance calls for overlapping views, useful texture, similar lighting
and actual viewpoint changes; it recommends downsampling redundant video frames.
Its SfM stage estimates cameras and sparse structure, and MVS recovers denser
geometry. A camera reconstruction alone does not produce a human animation rig.
Source: [COLMAP tutorial](https://colmap.github.io/tutorial.html).

A compilation with cuts, different clothing, changing lighting and moving hands
should not be treated as synchronized multiview capture. For the first trial,
select a continuous sequence and review visibility of face, shoulders, elbows,
wrists and fingers. Additional photographs can guide likeness and texture edits
without being assumed compatible with a single-video optimizer.

## ExAvatar: expressive reconstruction from a coherent video

The [official project](https://mks0601.github.io/ExAvatar/) demonstrates body, hand
and facial-expression control after reconstruction from a short phone scan,
approximately ten seconds in a neutral pose. It combines SMPL-X deformation with
Gaussian appearance and uses surface connectivity to regularize deformation.
This is evidence of expressive reconstruction, not a benchmark on edited footage.

The [released fitting pipeline](https://github.com/mks0601/ExAvatar_RELEASE/blob/main/fitting/README.md)
fits a monocular video before avatar training. It recommends frames containing
most of the person without truncation, supports a moving-camera COLMAP path and
a static-camera virtual-camera path, and separates training/test frames. The
example subsamples a 30 fps source to 5 fps for avatar training/evaluation.
Upper-body-only source support is not established by these instructions.

**Availability:** the [official release repository](https://github.com/mks0601/ExAvatar_RELEASE)
contains fitting and avatar code. Input preparation still requires separate
models and preprocessing tools; this is not a single downloadable GLB generator.
Its [environment file](https://github.com/mks0601/ExAvatar_RELEASE/blob/main/environment.yml)
describes a Linux/CUDA-oriented stack. Native macOS/MPS support and a minimum
end-to-end VRAM requirement were not established by this review.

**Integration gap:** retain the Gaussian renderer and pose/expression parameters
for the experiment. No verified facial-morph GLB/FBX export or ready-made mapping
from new speech audio was found in the inspected workflow. Evaluate novel mouth
shapes, eye closure and individual finger motion separately from neutral renders.

**Licensing:** own [code is MIT](https://github.com/mks0601/ExAvatar_RELEASE/blob/main/LICENSE).
The fitting instructions require **SMPL-X 1.1 and FLAME 2019/2020**, whose model
licenses are separate, plus a Depth Anything V2 Large checkpoint. Their terms are
listed under [License boundaries](#license-boundaries); MIT does not cover them.

## LHM++: pose-free multi-photo Gaussian experiment

The [official README](https://github.com/aigc3d/LHM-plusplus/blob/main/README.md)
describes reconstruction from one or multiple pose-free images. This makes it a
candidate for selected photographs without supplied camera/pose annotations.
It does not establish reliable identity recovery across arbitrary ages, clothing
changes, occlusions or inconsistent sources.

The model table reports about **8 GB** for listed 700M configurations. This is an
upstream model requirement, not a measured peak for preprocessing, training or
this repository. The [installation guide](https://github.com/aigc3d/LHM-plusplus/blob/main/INSTALL.md)
targets tested Ubuntu/Linux, Python 3.10, PyTorch 2.3 and CUDA 12.1, with compiled
GPU dependencies. Native MPS operation has not been verified.

**Checkpoint caveat:** at inspection, the README marked default
`LHMPP-700M-PixelShuffle` Hub weights **pending**, requiring a local checkpoint.
The updated-model announcement and supported-export table are not fully aligned.
Record the exact available checkpoint, revision and selected renderer before a
reproduction attempt; a default command is not proof that its weights are public.

**Output:** documented `to_gs_ply.py` exports Gaussian PLY in a canonical pose or
one supplied SMPL-X pose. The documented export variants are PixelShuffle and
SMPLX-FREE. The output is not a portable skinned mesh with facial morph targets.
Source: [README export instructions](https://github.com/aigc3d/LHM-plusplus/blob/main/README.md).

The [skinning source](https://github.com/aigc3d/LHM-plusplus/blob/main/core/models/rendering/skinnings/smplx_voxel_skinning.py)
contains hand, jaw and expression handling. Calling the method strictly
"body-only" would be inaccurate. Source-level expression support does not prove
speech-quality facial animation or compatibility with this runtime's face names.

**Licensing:** top-level [Apache 2.0](https://github.com/aigc3d/LHM-plusplus/blob/main/LICENSE),
[CC BY-NC 4.0 weights](https://github.com/aigc3d/LHM-plusplus/blob/main/LICENSE_WEIGHT),
and separately included [NVIDIA EG3D terms](https://github.com/aigc3d/LHM-plusplus/blob/main/LICENSE_NVIDIA)
must be recorded independently. The covered NVIDIA code has a noncommercial
research/evaluation limitation. "SMPLX-FREE" is not a blanket license clearance.

## MPFB / Blender: controllable mesh foundation

MPFB provides a human mesh and rigging/shape authoring tools. It is a foundation
for fitting/sculpting a new likeness; it is not an automatic photo reconstruction
service. A successful body setup does not settle face likeness, hair, clothing,
skin texture or facial deformation quality.

[MPFB 2.0.16](https://www.makehumancommunity.org/mpfb/releases/release_2016.html)
adds a shape-key expression workflow using the separately installed `faceunits01`
pack. Use the modern **add Rigify meta rig, then generate** workflow when choosing
Rigify; the legacy **Convert to Rigify** route explicitly lacks a face rig.
The [Blender extension history](https://extensions.blender.org/add-ons/mpfb/versions/)
lists 2.0.16, released 2026-06-13, for Blender 4.2 LTS and newer.

[Export-copy documentation](https://static.makehumancommunity.org/mpfb/docs/exporting/export_copy.html)
describes Microsoft-style visemes, Meta-style visemes and ARKit-style face-unit
shape keys, with optional propagation to affected child meshes. Packs must be
installed. Upstream pages differ on the face-unit count; inspect actual exported
target names instead of asserting that a particular count proves compatibility.

[Finger helpers](https://www.makehumancommunity.org/mpfb/docs/rigging_posing/rig_helpers.html)
provide per-finger IK pointing and individual/master grip controls on supported
default rigs. These authoring controls still need usable deform bones and weights
in the exported asset; Blender control widgets are not the runtime contract.

The [official extra-targets repository](https://github.com/makehumancommunity/extra-targets)
describes its generated visemes as needing visual polish. Retest blinks, lip
closure, jaw opening and cheeks after likeness edits or topology changes.
The [MPFB license](https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md)
separates GPLv3-or-later program code from CC0 bundled graphical assets. Record
licenses for additional hair, clothing and textures individually.

## Anny: optional differentiable fitting component

[Anny](https://github.com/naver/anny) is a PyTorch parametric human mesh model,
not a turnkey photo-to-textured-avatar application. Its README lists **v0.6.2
(2026-10-26)**. The native default rig has 104 bones for body, hand and head
articulation; facial-action blendshapes are enabled separately. The fuller
MakeHuman rig retains face/eye/tongue bones omitted from the compact rig.

Proposed use: optimize shape/pose against selected landmarks, silhouettes or
reconstructed geometry, then refine the mesh and materials in Blender. Designing
that fitting objective and validating subject likeness are project work. A
parametric mesh export does not demonstrate a complete textured GLB exporter.

The [code license](https://github.com/naver/anny/blob/main/LICENSE) is Apache 2.0.
The [README license section](https://github.com/naver/anny#license) identifies CC0
MakeHuman/MPFB-derived assets and face units, while optional SMPL-X interoperability
assets are noncommercial. It warns that even a free install may download such
assets when needed. Pin native topology/rig choices and log downloaded assets.
No native MPS fitting pipeline or end-to-end GPU-memory budget was verified here.

## MetaHuman: optional complete mesh-to-character workflow

[MetaHuman 5.8 release notes](https://dev.epicgames.com/documentation/metahuman/metahuman-5-8-release-notes-in-unreal-engine)
confirm support for arbitrary-topology head and/or body meshes. The result uses
MetaHuman topology and rig; this does not reconstruct identity directly from an
unorganized photo collection or consume Gaussian PLY as a rigged human mesh.

The [From Custom Mesh workflow](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-from-custom-mesh-tool-in-unreal-engine)
works best with realistic morphology and an A-pose. Fingers/toes may require
manual keypoint correspondence. Loose clothing is fitted as part of the surface,
so body fitting, garment transfer and texture baking require separate decisions.

**Export distinction:** `Save Pose` creates intermediate DNA containing body and
head on the body skeleton **without the facial rig**. It is for the Maya Pose
Editor and cannot be passed to Character Assembler as a complete character.
Proceed through Creator **Assembly** and a complete DCC export for the full
character workflow. A subsequent GLB conversion, face controls, materials and
browser performance still require validation in this project.

[Current licensing](https://www.metahuman.com/license) permits use in other engines
and creative software under Unreal terms. It is not a blanket open-source asset
license. Record the selected tool version, platform requirements and export path;
do not infer that a Blender workflow or PyTorch MPS support applies to MetaHuman.

## Charakuru: Japanese staged-automation reference

[Charakuru / きゃらくる](https://github.com/nanocle/Charakuru) packages an
illustration-to-VRM workflow using Tripo, Blender, an AI agent and user review.
Its public repository explains the workflow; downloadable kit releases are
separate from GitHub's automatic source archive. This is useful as an example of
staged generation, rigging, expression work and motion review, not evidence of
photoreal identity reconstruction from a real person's photographs.

The [release page](https://github.com/nanocle/Charakuru/releases) lists **1.1.1
(2026-11-02)**, with changes to upper-chest support, finger weights/rest pose and
expression transfer. These are upstream release claims, not locally verified
quality results. Service accounts, external generation terms and costs remain
separate from the kit.

Its [custom source-available license](https://github.com/nanocle/Charakuru/blob/main/LICENSE)
is explicitly not MIT or an OSI-approved open-source license. Business-use
eligibility and redistribution conditions apply; bundled donor assets cannot be
extracted for independent reuse. This plan borrows the idea of staged review and
does not propose copying kit code, donor meshes or assets into this repository.

## License boundaries

| Layer | Verified source and implication |
| --- | --- |
| ExAvatar implementation | [MIT code](https://github.com/mks0601/ExAvatar_RELEASE/blob/main/LICENSE); downloaded models retain their own terms. |
| SMPL-X assets | [Model license](https://smpl-x.is.tue.mpg.de/modellicense.html) limits the standard grant to specified noncommercial uses and restricts distribution; separate commercial licensing exists. |
| FLAME versions used by ExAvatar | [Model license](https://flame.is.tue.mpg.de/modellicense.html) covers 2019/2020 under noncommercial terms. A differently licensed newer model is not a verified drop-in replacement. |
| ExAvatar depth checkpoint | [Fitting instructions](https://github.com/mks0601/ExAvatar_RELEASE/blob/main/fitting/README.md) select `depth_anything_v2_vitl.pth`; [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2#license) licenses Large weights under CC BY-NC 4.0. |
| MPFB / Anny assets | The linked upstream notices distinguish code from CC0 base assets; optional third-party downloads can introduce different terms. |
| LHM++ | Apache code, noncommercial weights and separately covered NVIDIA code must remain distinct in any dependency inventory. |
| MetaHuman / Charakuru | Follow their tool, asset and redistribution terms; free access or visible source does not establish unrestricted redistribution. |

This is a scoped technical license inventory, not an exhaustive dependency audit
or clearance of source photographs, footage, textures or resulting likenesses.

## Evidence required before adopting a method

Record the upstream commit/release, checkpoint URL and hash, model-data versions,
licenses, environment, input manifest, output representation and actual peak
memory/time. Separate demonstrations from reproduced results and untested assumptions.

Use common development views and motion sequences for method selection, and
reserve final event-level holdouts for the frozen candidate's qualification, as
specified in the [data protocol](../avatar-generation-pipeline.md#32-coverage-and-preparation-output).
Evaluate:
identity from multiple angles; torso/arm continuity; shoulder/elbow/wrist motion;
finger separation/grips; blink and lip closure; mouth interior; texture stability;
and the actual exported asset under target-runtime lighting and camera framing.
New speech needs its own timed expression/viseme mapping, and gestures need their
own motion track. SMPL-X coefficients, ARKit-style targets and rig controls must
be mapped explicitly; shared labels do not prove interchangeable deformation.
