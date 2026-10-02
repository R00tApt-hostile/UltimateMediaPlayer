# Ultimate Media Player

A desktop media player built with PyQt5. Plays audio and video, shows a live audio visualizer, scans a music folder into a library, and pulls in lyrics when it can find them.

I wrote this to scratch an itch: I wanted a player that wasn't bloated, that I could actually read the source of, and that showed me what the music was doing instead of just a progress bar. The current state is functional and reasonably stable. It's not VLC and it never will be, but it does what it says.

## What it does

- Plays MP3, WAV, OGG, FLAC, M4A, AAC, WMA, plus MP4, AVI, MKV, MOV, WMV, FLV
- Live FFT-based visualizer with a few different render modes (spectrum, waveform, spectrogram, bars, particles)
- Scans a folder in the background and builds a library with metadata (title, artist, album, duration, bitrate) via mutagen and eyed3
- Reads embedded lyrics from ID3 USLT frames, and falls back to lyrics.ovh / AZLyrics if the tag is empty
- Eight color themes you can swap at runtime
- System tray with playback controls
- Keyboard shortcuts for the stuff you actually do often
- Remembers your window layout, volume, theme, and recent files between sessions
- Sleep timer

## Why this isn't just a copy-paste dump

This started life as a partially-written sketch. A lot of it didn't compile and a lot of it didn't work. Getting it to a runnable state meant fixing a stack of real bugs. If you're reading the code and wondering why certain things look defensive, this is why:

| Problem | What I did |
|---------|-----------|
| `AudioVisualizer` was referenced but never defined. The app crashed on the first line of `__init__`. | Wrote the class. |
| About fifteen widgets (`play_button`, `progress_slider`, `visualizer_button`, etc.) were used in slots but never instantiated. Every signal handler was an `AttributeError` waiting to happen. | Constructed them all and wired them up properly. |
| `AudioAnalyzer` was killed permanently by `stop_analysis()`. Calling `start()` again raised `RuntimeError` because QThread can't be restarted after it finishes. | Rewrote it with a `QWaitCondition` and proper pause/resume/stop semantics. The thread is now reusable. |
| The code connected to `QMediaPlaylist.loaded`, which is not a real signal. Hard crash on startup. | Removed. Only valid Qt signals are connected now. |
| `self.recent_files` was appended to before it was ever initialized. | Loaded from `QSettings` in `load_settings()`. |
| `setup_menus`, `setup_toolbars`, `setup_controls`, `setup_progress_controls`, and `setup_status_bar` were all empty. The menu bar was literally blank. | Implemented them. |
| `toggle_mute()` could access `self.saved_volume` before it existed, because the attribute was only set on the muted branch. | Initialized in `__init__`. |
| The `youtube_dl` import was dead. That project was abandoned years ago and fails on modern Python. | Swapped to `yt_dlp`, and made it optional. |
| The AZLyrics parser searched for `div class=None`, which matches nothing. | Rewrote the traversal. Added a User-Agent because AZLyrics blocks the default one. |
| `QThread.terminate()` was being called on the lyrics fetcher. That's unsafe in Qt — it can corrupt internal state and take the app down with it. | Cooperative cancellation via a flag. |
| `QSettings.value(..., type=bool)` doesn't do what the original author thought it does. | Explicit string comparison. |
| Directory scanning ran on the GUI thread. Point it at a folder with ten thousand files and the window froze until it finished. | Moved it to a `LibraryScanner(QThread)`. |
| The audio ring buffer used `np.roll`, which silently dropped samples and duplicated others on every roll. The FFT was analyzing garbage. | Replaced with a proper linear shift of the remaining samples. |
| `cdrom`, `sounddevice`, `pydub`, `librosa`, and `PIL` were all imported and none of them were used. | Removed. |
| The `sys.excepthook` referenced `player` before it was safely bound, so errors during startup crashed the crash handler. | Guarded it. |

None of this is glamorous work. It's just what had to happen to get the thing to run.

## Requirements

```
pip install PyQt5 numpy mutagen eyed3 requests beautifulsoup4
```

For the (currently scaffolded) streaming path:

```
pip install yt-dlp
```

Python 3.8 or newer. Tested on Windows and Linux. Should work on macOS but I haven't tried it recently.

On Linux you may need to install `python3-pyqt5.qtmultimedia` from your distro's package manager. The multimedia module isn't always bundled with the base PyQt5 package.

## Running it

```
python ultimate_media_player.py
```

### Shortcuts

| Key | Does |
|-----|------|
| Space | Play / pause |
| Ctrl+. | Stop |
| Ctrl+Left / Ctrl+Right | Seek back / forward 5 seconds |
| Ctrl+M | Mute |
| Ctrl+O | Open files |
| Ctrl+Q | Quit |
| F11 | Fullscreen |
| Esc | Leave fullscreen |

### Layout

The media library lives in a dock on the left. The playlist and lyrics panels are on the right. You can close any of them from the View menu, and the app will remember what you had open. The transport controls, volume slider, and visualizer mode selector are at the bottom.

To populate the library, use File → Open Folder and point it at your music directory. The scan runs in the background, so you can keep using the player while it works.

## How it's put together

```
UltimateMediaPlayer (QMainWindow)
├── AudioAnalyzer         (QThread)  — FFT, RMS, peak
├── LibraryScanner        (QThread)  — background directory walk
├── LyricsFetcher         (QThread)  — lyrics.ovh, then AZLyrics
├── AudioVisualizer       (QWidget)  — paints the spectrum
├── MediaLibrary                     — metadata cache
└── Qt Multimedia
    ├── QMediaPlayer + QMediaPlaylist
    ├── QAudioProbe → AudioAnalyzer.process_audio()
    └── QVideoWidget
```

Everything that takes time runs off the GUI thread. The threads pause and resume cleanly and shut down in `closeEvent` without leaking.

## Themes

Switch from View → Theme. The choice persists. There are eight: Dark, Light, Midnight, Professional, and four tinted variants (Blue, Green, Red, Purple). They're just palette swaps. If you want to add one, the switch is in `apply_theme()`.

## What's not done

Being honest about the gaps:

- **Equalizer.** There's no EQ. `QAudioProbe` is read-only — you can watch the audio but you can't touch it. A real EQ needs a DSP backend that intercepts the stream, and that's a much bigger project than this. Stubbed and disabled.
- **CD ripping.** Scaffolded with a placeholder thread that simulates progress. Wiring it up needs `cdparanoia` or `libcdio` and platform-specific code I haven't written.
- **Streaming.** `yt-dlp` is imported and there's a config dict, but nothing is hooked into the UI yet.
- **Gapless playback.** Not supported by the Qt Multimedia backend, as far as I can tell. Not going to fake it.
- **Subtitles.** Not implemented for video.
- **Playlist import/export.** No `.m3u` or `.pls` support yet.
- **ReplayGain.** No volume normalization.
- **Last.fm.** No scrobbling.

None of these are hard problems, they're just not solved here yet.

## Contributing

If you want to fix something or add a feature, open an issue first so we can talk about it. Then send a PR. Small, focused changes are easier to review than big ones. Include a note about what you tested on.

## A few caveats

Metadata reading is best-effort. Files with missing or malformed tags fall back to the filename and "Unknown" instead of crashing. That's deliberate.

Lyrics fetching depends on two services that can change their HTML whenever they want. The parsers are written defensively but they will break eventually and I'll have to fix them. If lyrics stop working, that's probably why.

The original version of this file was a non-functional sketch, most of which never ran. What you're looking at has been substantially rewritten. I kept the feature set and the overall shape, but the implementation is mostly new.