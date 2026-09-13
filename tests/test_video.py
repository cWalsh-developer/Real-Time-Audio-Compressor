import subprocess

import imageio_ffmpeg

from adaptive_audio.video import process_video


def test_mov_processing_keeps_video_stream_and_replaces_audio(tmp_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source = tmp_path / "source.mov"
    target = tmp_path / "target.mov"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=64x64:r=24:d=2", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=44100:duration=2", "-c:v", "mpeg4",
        "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    process_video(source, target, "balanced")
    assert target.is_file() and target.stat().st_size > 0
    def video_hash(path):
        result = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", "0:v:0", "-c", "copy", "-f", "hash", "-hash", "SHA256", "-",
        ], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    assert video_hash(source) == video_hash(target)
    result = subprocess.run([ffmpeg, "-hide_banner", "-i", str(target)], capture_output=True, text=True)
    assert "Audio: aac" in result.stderr
