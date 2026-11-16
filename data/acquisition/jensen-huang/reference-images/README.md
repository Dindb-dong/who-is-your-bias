# Jensen Huang 3D Avatar Reference Dataset

Reference image set for 3D face/body reconstruction and manual modeling.
Collected on 2026-11-12. Only sources with clear licenses (Wikimedia Commons, CC-licensed Flickr via Openverse) were used.

## Summary

| Item | Count |
|---|---|
| Unique source images | 81 (71 stills + 10 video frames) |
| Total files (including face crops) | 105 (+ 28 copies in `face/expressions`) |
| Candidates reviewed | 171 (Commons 78 + Flickr/Openverse 93) |
| Excluded: other people / not verified | 81 |
| Excluded: duplicates (crops, re-uploads) | 13 |
| Excluded: quality issues (screen captures, occlusion, blur) | 6 |

| Folder | Files | Contents |
|---|---|---|
| `face/frontal` | 28 | \|yaw\| < 15° |
| `face/left_45` | 22 | 15° ≤ \|yaw\| < 60°, face pointing toward the left of the frame |
| `face/right_45` | 14 | 15° ≤ \|yaw\| < 60°, face pointing toward the right of the frame |
| `face/left_profile` | 2 | \|yaw\| ≥ 60°, face pointing toward the left of the frame |
| `face/right_profile` | 0 | (none found in openly licensed sources, see below) |
| `face/expressions` | 28 | Copies of smiling face images (`smiling_*.jpg`) |
| `upper_body` | 28 | Upper-body originals |
| `full_body` | 11 | Full-body originals |

## Conventions

- **Left/right is from the viewer's side (the image).** `left_*` = nose points toward the left of the frame (yaw < 0). In terms of the subject's own body, he has turned to his right.
- `viewing_angle` suffix: `_up` (looking up, pitch < -10°), `_down` (looking down, pitch > 10°). The raw values are in `yaw_deg` / `pitch_deg`.
- `*_face.jpg` = a crop (2.2× the face box) around Jensen's face from a group or wide shot. `parent_image_id` points to the source image. No crops were made from images with no-derivatives (ND) licenses.
- For group/wide shots where the body isn't a useful reference, only the face crop is kept as a file; the source is recorded in the URL fields.

## How the dataset was built

1. **Identity check**: OpenCV YuNet (face detection) + SFace (face recognition). The reference embedding is the average of 2 clear portraits. Faces with cosine similarity ≥ 0.45 are kept, and every candidate was checked visually on contact sheets. Real matches scored 0.5–0.88; other people scored ≤ 0.36.
2. **Angle and expression**: yaw/pitch from MediaPipe FaceLandmarker's transformation matrix; expression from blendshapes (mouthSmile, jawOpen). 4 images where detection failed (profiles, etc.) were tagged by eye (`identity_score` empty or yaw at a round value).
3. **Framing**: whether the ankles/hips/shoulders are visible in MediaPipe PoseLandmarker's result.
4. **Deduplication**: same photo if aligned face embedding similarity ≥ 0.88 **and** pixel NCC ≥ 0.70. Only the highest-resolution copy is kept; removed files are listed in `duplicates_removed`.
5. **Video frames**: from a 104-second CC BY 3.0 video on Commons, sampled every 0.5 s. The sharpest frame was kept for each yaw (10° bins) × pitch × expression combination, at least 1.5 s apart.

The code is in `_pipeline/` (run order: commons_list → download → analyze → video_frames → organize).

## Resolution note

`upload.wikimedia.org` blocked original downloads with HTTP 429 (rate limiting), so Commons images were downloaded at the largest standard thumbnail size (3840/1920/1280px) that doesn't exceed the original width. `resolution` is the saved file's size, `original_resolution` is the original's size, and `image_url` is the original URL. If you need originals, download them from `image_url` later.

## License and usage (important)

The `usage` column in `metadata.csv`:

- `OPEN: attribution required` — CC BY / CC BY-SA / GODL-India / OGL. Credit the author and license when you use or publish (SA: derivatives must use the same license).
- `OPEN: no copyright restriction` — US government works (White House photos, etc.), CC0, PDM.
- `LIMITED: non-commercial` — CC BY-NC*. Fine for an academic project, but **must be excluded if the project becomes commercial**.
- `RESTRICTED: no-derivatives` — CC BY-NC-ND (2 images). Use only as viewing reference; don't modify or redistribute.

**Separate from copyright**, using a real person's likeness and name involves **publicity/personality rights** and trademarks (the NVIDIA logo). Even with a parody concept, consider stating that it isn't official in public demos or releases, and avoid commercial use and wording that could be mistaken for the real person's statements. Before pushing this dataset to a public GitHub repo, check the license conditions (attribution notice, NC/ND). A safer option is to commit only `metadata.csv` and the scripts.

## internal_reference/ (added 2026-11-13, copyrighted, internal use only)

107 extra source images and 179 files from official NVIDIA material (press kits, blog, press conference video). They have no open license, so they're kept in a separate folder with a separate `metadata_internal.csv`. **Never push this folder to GitHub.** Details are in `internal_reference/README.md`.

## Known gaps

- **Profiles are lacking**: left_profile has 2, right_profile has 0. Openly licensed sources have almost no profile shots. For full side reference you'll need to add more, for example from frames you record yourselves or press photos whose terms you've confirmed.
- **Limited expression variety**: blendshape classification only produced neutral and smiling; laughing/speaking couldn't be separated reliably. Some of the video frames are mid-speech.
- **Looking up/down**: up 4, down 11 (mostly slight tilts).
- Some face crops are low resolution (150–300px). Check the `face_size_px` column and filter for your purpose.
