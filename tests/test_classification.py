import json

import pytest

from adaptive_audio.classification import EventScores, ManualClassifier, read_scores


def test_manual_classifier_returns_scores_for_each_time_window(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"segments": [
        {"start_seconds": 0, "end_seconds": 1, "scores": {"speech": 0.9, "music": 0.1, "action": 0, "other": 0}},
        {"start_seconds": 1, "end_seconds": 2, "scores": {"speech": 0, "music": 0, "action": 0.8, "other": 0.2}},
    ]}))
    classifier = ManualClassifier.from_json(labels)
    scores = classifier.classify_duration(3, window_seconds=1)
    assert [item.scores["speech"] for item in scores] == [0.9, 0, 0]
    assert scores[2].scores["other"] == 1
    assert [(item.start_seconds, item.end_seconds) for item in scores] == [(0, 1), (1, 2), (2, 3)]


def test_invalid_scores_are_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        EventScores(0, 1, {"speech": 0.8, "music": 0.8, "action": 0, "other": 0})


def test_overlapping_manual_segments_are_rejected(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"segments": [
        {"start_seconds": 0, "end_seconds": 1.2, "scores": {"speech": 1, "music": 0, "action": 0, "other": 0}},
        {"start_seconds": 1, "end_seconds": 2, "scores": {"speech": 0, "music": 1, "action": 0, "other": 0}},
    ]}))
    with pytest.raises(ValueError, match="overlap"):
        ManualClassifier.from_json(labels)


def test_read_scores_accepts_overlapping_model_windows(tmp_path):
    labels = tmp_path / "model.json"
    labels.write_text(json.dumps({"segments": [
        {"start_seconds": 0, "end_seconds": 0.96, "scores": {"speech": 1, "music": 0, "action": 0, "other": 0}},
        {"start_seconds": 0.48, "end_seconds": 1.44, "scores": {"speech": 0, "music": 1, "action": 0, "other": 0}},
    ]}))
    assert len(read_scores(labels)) == 2
