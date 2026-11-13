"""Invented miniatures test parsing only; never count these as source dialogue."""
import json
from html import escape

from tools.persona_data.jensen import colon_turns, dwarkesh, goldman, strange_loop, yc
from tools.persona_data.parsers import html_blocks


def test_contiguous_speaker_paragraphs_and_unknown_label_barrier():
    blocks = html_blocks(b'<p>Host: Question?</p><p>Jensen: Original answer.</p><p>Qualification stays.</p><p>Other Guest: Not Jensen.</p><p>Unknown continuation.</p><p>Host: Next?</p><h2>Footer</h2><p>Jensen: Not body.</p>')
    rows = colon_turns(blocks, {'Host': 'Host', 'Jensen': 'Jensen Huang'})
    assert rows[1]['text'] == 'Original answer. Qualification stays.'
    assert rows[2]['speaker'] is None
    assert 'Unknown continuation' not in ' '.join(x['text'] for x in rows if x['speaker'])
    assert not any('Not body' in x['text'] for x in rows)


def test_dwarkesh_jensen_names_and_chapter_attribution():
    rows = dwarkesh(b'<h2>Transcript</h2><p><strong>Dwarkesh Patel</strong></p><p>Question?</p><p><strong>Jensen Huang</strong></p><p>Answer.</p><h3>Chapter</h3><p>No label.</p><p><strong>Jensen Huang</strong></p><p>New answer.</p><h3>Ready for more?</h3><p>Tags.</p>')
    assert rows[1]['speaker'] == 'Jensen Huang'
    assert not any('No label' in r['text'] or 'Tags.' in r['text'] for r in rows)


def test_strange_loop_retains_literal_clock_and_removes_terminal_tags():
    data = b'<p>Joel | 00:01:02:33 - 00:01:08:12</p><p>Question?</p><p>Jensen | 00:01:08:17 - 00:02:10:11</p><p>Answer. AI - Multimodality AI - physical feedback model more tags</p><p>Cookies</p><p>Footer.</p>'
    rows, audit = strange_loop(data)
    assert rows[1]['text'] == 'Answer.'
    assert rows[1]['start'] is None
    assert rows[1]['publisher_locator'] == '00:01:08:17 - 00:02:10:11'
    assert audit[0]['reason'] == 'publisher_trailing_topic_tags'
    assert audit[0]['before_sha256'] != audit[0]['after_sha256']


def test_yc_embedded_original_labels_only():
    data = {'props': {'article': {'transcript': 'Garry Tan (host): A question?\n\nJensen Huang (guest): An answer.\nMore answer.'}}}
    raw = ('<div data-page="' + escape(json.dumps(data), quote=True) + '"></div>').encode()
    rows = yc(raw)
    assert [r['speaker'] for r in rows] == ['Garry Tan', 'Jensen Huang']
    assert rows[1]['text'] == 'An answer. More answer.'


def test_goldman_discards_legal_footer_without_deleting_spoken_bank_name():
    rows = goldman('Goldman Sachs Talks\nDavid Solomon: Question?\nJensen Huang: We work with\nGoldman Sachs on this.\n2\nThank you.\nThe opinions and views expressed in this program may not\nreflect the firm.')
    assert rows[-1]['text'] == 'We work with Goldman Sachs on this. Thank you.'


def test_yc_mixed_guest_typography_does_not_merge_questions_and_answers():
    data = {'transcript': 'Garry Tan (host): First?\n\nJensen Huang (guest): First answer.\n\nGarry Tan (host): Second?\n\nJensen Huang: Second answer.\n\nGarry Tan: Third?\n\nJensen Huang (guest): Third answer.'}
    raw = ('<div data-page="' + escape(json.dumps(data), quote=True) + '"></div>').encode()
    rows = yc(raw)
    assert [r['speaker'] for r in rows] == ['Garry Tan', 'Jensen Huang'] * 3
    assert rows[3]['text'] == 'Second answer.'
    assert all('Jensen Huang:' not in r['text'] for r in rows)
