"""Bind MediaPipe landmarks to the rendered generic template (no identity claim)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path); parser.add_argument('model', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():raise ValueError('Template measurements are immutable')
    import mediapipe as mp
    options = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),num_faces=1)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result=detector.detect(mp.Image.create_from_file(str(args.image)))
        if len(result.face_landmarks)!=1:raise ValueError('Template neutral face was not detected')
        value={'schema_version':'avatar-template-landmarks.v1','image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
               'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
               'landmarks':[[p.x,p.y,p.z] for p in result.face_landmarks[0]]}
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(value,allow_nan=False)+'\n');temp.replace(args.output)
        print(json.dumps({'points':len(value['landmarks'])}))


if __name__=='__main__':main()
