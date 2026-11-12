"""Plot saved browser observations; this command does not run inference or playback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=Path('reports/jensen-runtime-v1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    reports = [json.loads((args.reports/f'recorded-browser-{name}.json').read_text())
               for name in ['desktop', 'mobile']]
    if any(r['schema_version'] != 'recorded-generated-audio-browser.v1' for r in reports):
        raise ValueError('Expected actual recorded-audio browser observations')
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex='col', constrained_layout=True)
    colors = {'jaw':'#006e90', 'blink':'#b24185', 'smile':'#578400',
              'arm':'#006e90', 'hand':'#d66700', 'finger':'#7951a4'}
    for column, report in enumerate(reports):
        frames = [x for x in report['trace'] if x['speaking'] and 100 < x['audioMs'] < 9000]
        seconds = [x['mediaMs']/1000 for x in frames]
        for row, names in enumerate([['jaw', 'blink', 'smile'], ['arm', 'hand', 'finger']]):
            for name in names:
                axes[row, column].plot(seconds, [x[name] for x in frames], label=name,
                                       color=colors[name], linewidth=1.4,
                                       marker='.' if column == 0 else None, markersize=3)
            axes[row, column].legend(loc='upper right', ncol=3, fontsize=8)
        gaps = [abs(x['audioMs']-x['mediaMs']) for x in frames]
        axes[2, column].plot(seconds, gaps, color='#006e90', linewidth=1.4)
        axes[2, column].axhline(750, color='#9b3333', linestyle='--', linewidth=1,
                               label='750 ms test gate')
        axes[2, column].legend(loc='upper left', fontsize=8)
        axes[0, column].set_title(f"{report['project'].title()} · {report['playing_frames']} observed playing frames")
        for row, axis in enumerate(axes[:, column]):
            axis.set_xlim(0, 9.92)
            axis.set_ylim(0, [1, .45, 820][row])
            stop = report['media_ms_before_stop']/1000
            axis.axvline(stop, color='#777777', linewidth=.8)
            axis.axvspan(stop, 9.92, color='#eeeeee', alpha=.6)
            axis.grid(alpha=.18)
        axes[0, column].text(8.7, .48, 'Stopped\nno samples', ha='center', fontsize=8, color='#666666')
        axes[2, column].set_xlabel('Actual media clock (seconds)')
    for row, label in enumerate(['Mapped morph weight', 'Joint angle\n(radians from neutral)', 'Absolute renderer/media gap\n(ms)']):
        axes[row, 0].set_ylabel(label)
    fig.suptitle('Recorded generated audio on the fitted Jensen candidate\nActual mapped outputs · authored gesture cues · no live inference or acoustic latency measurement', fontsize=12)
    output = args.output or args.reports/'renders/recorded-browser-trace.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
