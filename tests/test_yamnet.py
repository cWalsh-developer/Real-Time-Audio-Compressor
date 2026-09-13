import numpy as np

from adaptive_audio.yamnet import group_scores


def test_group_scores_maps_speech_music_and_explosion():
    names = ["Speech", "Music", "Explosion", "Silence"]
    speech = np.array([0.9, 0.1, 0.05, 0.02])
    music = np.array([0.1, 0.8, 0.05, 0.02])
    action = np.array([0.05, 0.05, 0.95, 0.02])
    assert max(group_scores(speech, names), key=group_scores(speech, names).get) == "speech"
    assert max(group_scores(music, names), key=group_scores(music, names).get) == "music"
    assert max(group_scores(action, names), key=group_scores(action, names).get) == "action"
    assert max(group_scores(np.array([0.01, 0.01, 0.01, 1.0]), names), key=group_scores(np.array([0.01, 0.01, 0.01, 1.0]), names).get) == "other"
