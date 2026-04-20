# Karaoke Pitch Tracker

A desktop application that tracks your singing pitch in real-time and compares it to the original song, Rock Band 4-style. Build a catalog of songs from your Spotify playlists, or use Live Mode to sing along to anything playing on your computer. It runs on **Windows** and **macOS** (Linux may work with a manual loopback input, but is not the focus).

## Features

- **Song Catalog**: Import songs from Spotify playlists, automatically download audio, isolate vocals, and extract pitch maps
- **Rock Band-style Pitch Display**: Scrolling visualization showing target notes and your real-time pitch
- **Real-time Scoring**: Cents-based pitch accuracy scoring with combo tracking and star ratings
- **Live Mode**: Capture system / speaker mix audio and sing along to anything playing on the machine (WASAPI loopback on Windows; virtual loopback input such as **BlackHole** on macOS — see below)
- **Performance History**: Track your scores and improvement over time
- **Vocal Chain**: Live mic cleanup (high-pass + noise gate + optional Silero VAD), vocal monitor output (hear yourself sing), and an effects chain (HPF, 3-band EQ, compressor, reverb, gain) with presets — Natural, Warm, Bright, Stage Reverb, Studio Polish

## Vocal monitor and feedback

Turn on **Settings → Vocal monitor** to route your mic through the effects chain to a chosen output device. **Use headphones for the monitor**, separate from the backing-track output. If both share the same speakers you'll get a feedback loop; the built-in *Feedback protection* keeps the monitor muted whenever the resolved monitor device equals the backing-track device.

For the lowest latency on Windows, pick a **WASAPI** output device for the monitor — the app will automatically use WASAPI shared mode for tighter scheduling than MME / DirectSound.

### Live Mode on macOS

macOS does not expose a built-in “what you hear” loopback API like WASAPI. Use a free virtual audio device (commonly **[BlackHole](https://github.com/ExistentialAudio/BlackHole)**): in **Audio MIDI Setup**, create a **Multi-Output Device** that includes both BlackHole and your real speakers/headphones, set that aggregate as the system output for playback, then in **Live Mode** choose the **BlackHole** input under **System audio input** (likely shown with a ★ prefix).

## Requirements

- Python 3.11+
- **Windows 10/11** for zero-driver WASAPI loopback in Live Mode, **or macOS** with a virtual loopback input (e.g. BlackHole) as above
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

   **Choosing a GPU (CUDA).** After CUDA-enabled PyTorch and a working driver are installed, you can steer Demucs / torchcrepe without code changes by copying variables from `.env.example` into `.env` (loaded automatically when the app starts):
   - **`TORCH_DEVICE`** — explicit device when **Settings → Use GPU** is on, e.g. `cuda`, `cuda:0`, or `cuda:1`.
   - **`TORCH_CUDA_DEVICE_INDEX`** or **`CUDA_DEVICE_INDEX`** — use GPU *N* when `TORCH_DEVICE` is unset (handy on multi-GPU machines).
   - **`CUDA_VISIBLE_DEVICES`** — standard NVIDIA variable; only listed physical GPUs are visible (the first becomes logical `cuda:0`).

   Unchecking **Use GPU** in Settings forces **CPU** for processing regardless of the above. **Settings** also shows the resolved device string and any of these environment values.

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
| Ctrl+1 (Windows) / Cmd+1 (macOS) | Go to Catalog |
| Ctrl+2 / Cmd+2 | Go to Processing |
| Ctrl+3 / Cmd+3 | Go to Sing mode |
| Ctrl+4 / Cmd+4 | Go to Live Mode |
| Ctrl+5 / Cmd+5 | Go to Settings |
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
- PyAudioWPatch on Windows only (WASAPI loopback for system audio capture; omitted on macOS via `requirements.txt` markers)
- spotipy (Spotify API)
- yt-dlp (YouTube audio download)
- SQLite (local song catalog and performance history)
