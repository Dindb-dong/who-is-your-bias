from tools.motion.preview_selection import select_hands


def test_crossed_or_ambiguous_hands_are_not_invented_from_model_handedness():
    pose = {'landmarks': [{'x': .5, 'y': .5, 'visibility': 1}] * 33}
    hand = {'detection_index': 0, 'handedness': [{'label': 'Left', 'score': 1}],
            'landmarks': [{'x': .5, 'y': .5}] * 21}
    assert select_hands({'hands': [hand]}, pose) == {}
    pose['landmarks'] = [dict(p) for p in pose['landmarks']]
    pose['landmarks'][15]['x'] = .2
    hand['landmarks'] = [{'x': .2, 'y': .5}] * 21
    assert select_hands({'hands': [hand]}, pose) == {'left_hand_index': 0}
    second = {**hand, 'detection_index': 1}
    assert select_hands({'hands': [hand, second]}, pose) == {}
    hand['landmarks'] = [{'x': 1.1, 'y': .5}] * 21
    assert select_hands({'hands': [hand]}, pose) == {}
