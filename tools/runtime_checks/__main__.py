"""Launch the explicitly selected Jensen evidence-preview candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import managed_voice, prepare


def port(value: str) -> int:
    number = int(value)
    if not 1024 <= number <= 65535:
        raise argparse.ArgumentTypeError('Choose an unprivileged TCP port between 1024 and 65535')
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'serve', 'smoke'])
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--web-root', type=Path)
    parser.add_argument('--allow-candidates', action='store_true')
    parser.add_argument('--with-voice', action='store_true')
    parser.add_argument('--inference-python', type=Path)
    parser.add_argument('--voice-port', type=port, default=8092)
    parser.add_argument('--port', type=port, default=8000)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        settings, corpus, config, preflight = prepare(args)
        if args.command == 'verify':
            print(json.dumps(preflight, indent=2))
            return
        if args.command == 'smoke':
            if args.output is None:
                parser.error('smoke requires an external --output directory for original source excerpts')
            from .smoke import run
            result = run(args, settings, corpus, config, preflight)
            print(json.dumps(result, indent=2))
            raise SystemExit(0 if result['passed'] else 1)
        if args.port == args.voice_port and args.with_voice:
            parser.error('API and voice services require different ports')
        from bias.api import create_app
        import uvicorn
        with managed_voice(args, config):
            uvicorn.run(create_app(settings, corpus=corpus), host='127.0.0.1', port=args.port)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'Runtime setup failed: {exc}\n')


if __name__ == '__main__':
    main()
