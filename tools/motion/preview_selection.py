"""Build explicit, unreviewed per-frame selections for bounded geometry previews.

The operator supplies the face/pose detection indexes for an inspected window.
Multi-person frames are skipped. Handedness labels never identify the subject:
hand wrists must be unambiguously close to the selected pose's anatomical wrists.
This command cannot produce speaker_reviewed status or a biometric identity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tools.media.common import MediaError, local_file, sha256, write_json


def select_hands(row: dict, pose: dict, distance_limit: float = .07, ambiguity_margin: float = .012) -> dict:
    selected = {}
    landmarks = pose.get('landmarks', [])
    if len(landmarks) != 33:
        return selected
    for side, index in [('left', 15), ('right', 16)]:
        wrist = landmarks[index]
        if wrist.get('visibility', 0) < .65 or wrist.get('presence', 1) < .65:
            continue
        distances = []
        for hand in row.get('hands', []):
            points = hand.get('landmarks', [])
            if len(points) != 21:
                continue
            p = points[0]
            if not all(0 <= value <= 1 for value in [p['x'], p['y'], wrist['x'], wrist['y']]):
                continue
            distance = math.hypot(p['x'] - wrist['x'], p['y'] - wrist['y'])
            distances.append((distance, hand['detection_index']))
        distances.sort()
        if distances and distances[0][0] <= distance_limit and (len(distances) == 1 or distances[1][0] - distances[0][0] >= ambiguity_margin):
            selected[side + '_hand_index'] = distances[0][1]
    if selected.get('left_hand_index') is not None and selected.get('left_hand_index') == selected.get('right_hand_index'):
        return {}
    return selected


def make_preview(manifest_path: Path, output: Path, *, reviewer: str, track_id: str, face_index: int, pose_index: int) -> dict:
    if not reviewer.strip() or not track_id.strip() or face_index not in range(4) or pose_index not in range(4):
        raise MediaError('Explicit operator, track and frame-local indexes are required')
    manifest_path = local_file(manifest_path, max_bytes=1024**2)
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest['observations']
    observations = (manifest_path.parent / artifact['path']).resolve()
    if not observations.is_relative_to(manifest_path.parent.resolve()) or sha256(observations) != artifact['sha256']:
        raise MediaError('Observations differ from the supplied manifest')
    rows = [json.loads(line) for line in local_file(observations, max_bytes=256 * 1024**2).read_text().splitlines() if line.strip()]
    if not 1 <= len(rows) <= 3601:
        raise MediaError('Use one bounded observation window')
    selections = []
    for index, row in enumerate(rows):
        start = row['timestamp_seconds'] * 1000
        end = rows[index + 1]['timestamp_seconds'] * 1000 if index + 1 < len(rows) else start + 1000 / manifest['sample_fps_requested']
        selected = {'start_ms': start, 'end_ms': end, 'face_index': None, 'pose_index': None}
        faces, poses = row.get('faces', []), row.get('poses', [])
        # Never silently switch index when the selected person disappears, and
        # never infer a person from the detector's first result in a crowded frame.
        if len(faces) == 1 and faces[0]['detection_index'] == face_index:
            selected['face_index'] = face_index
            if len(poses) == 1 and poses[0]['detection_index'] == pose_index:
                selected['pose_index'] = pose_index
                selected.update(select_hands(row, poses[0]))
        selections.append(selected)
    result = {'schema_version': 'motion-review.v1', 'observations_sha256': artifact['sha256'],
              'status': 'geometry_preview', 'reviewer': reviewer, 'track_id': track_id,
              'track_continuity_reviewed': False, 'source_clock_reviewed': False, 'speaker_association_reviewed': False,
              'attribution_basis': 'Operator-selected window and detection indexes; frame-local hands associated by pose-wrist proximity. Multi-person or ambiguous detections are omitted. This is unreviewed geometry and does not verify speaker identity or continuous tracking.',
              'selections': selections}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--reviewer', required=True); parser.add_argument('--track-id', required=True)
    parser.add_argument('--face-index', type=int, required=True); parser.add_argument('--pose-index', type=int, required=True)
    args = parser.parse_args()
    result = make_preview(args.manifest, args.output, reviewer=args.reviewer, track_id=args.track_id, face_index=args.face_index, pose_index=args.pose_index)
    print(json.dumps({'status': result['status'], 'frames': len(result['selections']), 'hand_selected_frames': sum(any(row.get(side + '_hand_index') is not None for side in ['left', 'right']) for row in result['selections'])}))


if __name__ == '__main__':
    main()
