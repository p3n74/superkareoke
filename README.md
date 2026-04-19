# Karaoke Pitch Tracker

A Windows desktop application that tracks your singing pitch in real-time and compares it to the original song, Rock Band 4-style. Build a catalog of songs from your Spotify playlists, or use Live Mode to sing along to anything playing on your computer.

## Features

- **Song Catalog**: Import songs from Spotify playlists, automatically download audio, isolate vocals, and extract pitch maps
- **Rock Band-style Pitch Display**: Scrolling visualization showing target notes and your real-time pitch
- **Real-time Scoring**: Cents-based pitch accuracy scoring with combo tracking and star ratings
- **Live Mode**: Capture system audio (WASAPI loopback) and sing along to any audio playing on your PC
- **Performance History**: Track your scores and improvement over time

## Requirements

- Python 3.11+
- Windows 10/11 (WASAPI loopback requires Windows)
- A microphone
- (Optional) NVIDIA GPU with CUDA for faster processing
- (Optional) Spotify Developer credentials for playlist import
- (Optional) yt-dlp and FFmpeg installed and on PATH

## Installation

1. Clone or download this repository

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install PyTorch with CUDA (optional, for GPU acceleration). The default `pip install -r requirements.txt` often installs a **CPU-only** torch, so `torch.cuda.is_available()` stays false until you reinstall with a CUDA wheel. Use the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (pick your OS, CUDA version, and pip), for example:
   ```bash
   pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. Install yt-dlp and FFmpeg (required for downloading songs):
   ```bash
   pip install yt-dlp
   ```
   FFmpeg must be available on your system PATH. Download from https://ffmpeg.org/download.html

5. (Optional) Set up Spotify credentials:
   - Go to https://developer.spotify.com/dashboard and create an app
   - Copy `.env.example` to `.env` and fill in your credentials

## Usage

```bash
python main.py
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+1 | Go to Catalog |
| Ctrl+2 | Go to Processing |
| Ctrl+3 | Go to Sing mode |
| Ctrl+4 | Go to Live Mode |
| Ctrl+5 | Go to Settings |
| F11 | Toggle fullscreen |

### Workflow

1. **Add Songs**: Go to Catalog and click "+ Add Song" with the song title and artist
2. **Process**: Click "Process" on a song to download, isolate vocals, and extract pitch data
3. **Sing**: Once processing is complete, click "Sing!" to start performing
4. **Live Mode**: Or skip all that and use Live Mode to capture whatever is playing on your speakers

## Architecture

The app uses a pipeline architecture:

1. **Spotify API** (spotipy) fetches track metadata from your playlists
2. **yt-dlp** downloads the audio from YouTube
3. **Hybrid Demucs** (via torchaudio) separates the audio into vocals, drums, bass, and other
4. **torchcrepe** extracts the pitch contour from the isolated vocals
5. During performance, your microphone audio is analyzed in real-time with torchcrepe (tiny model) and compared to the pre-extracted pitch map

## Tech Stack

- PyQt6 (UI)
- PyTorch + torchaudio (Demucs vocal separation)
- torchcrepe (pitch detection)
- sounddevice (mic input and playback)
- PyAudioWPatch (WASAPI loopback for system audio capture)
- spotipy (Spotify API)
- yt-dlp (YouTube audio download)
- SQLite (local song catalog and performance history)
