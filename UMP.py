import os
import sys
import platform
import numpy as np
import traceback
from pathlib import Path
from collections import deque

from PyQt5.QtCore import (
    Qt, QUrl, QTimer, QSize, QPoint, QRect, QSettings, QStandardPaths,
    QFileInfo, QThread, pyqtSignal, QMutex, QMutexLocker, QWaitCondition,
)
from PyQt5.QtGui import (
    QIcon, QPalette, QColor, QPainter, QBrush, QPixmap, QFont, QPen,
    QLinearGradient, QKeySequence,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider,
    QLabel, QPushButton, QFileDialog, QListWidget, QListWidgetItem, QComboBox,
    QStyle, QSizePolicy, QFrame, QMessageBox, QMenu, QAction, QActionGroup,
    QSystemTrayIcon, QProgressDialog, QShortcut, QScrollArea, QDockWidget,
    QToolBar, QStatusBar, QInputDialog, QDialog, QDialogButtonBox, QFormLayout,
    QSpinBox, QCheckBox, QGroupBox, QTabWidget, QTextEdit, QLineEdit,
    QTreeWidget, QTreeWidgetItem, QStyleFactory,
)
from PyQt5.QtMultimedia import (
    QMediaPlayer, QMediaContent, QMediaPlaylist, QAudioProbe, QAudioBuffer,
)
from PyQt5.QtMultimediaWidgets import QVideoWidget

# Optional deps ---------------------------------------------------------------
try:
    import mutagen
    from mutagen.id3 import ID3
except ImportError:
    mutagen = None

try:
    import eyed3
except ImportError:
    eyed3 = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

try:
    import yt_dlp  # prefer yt-dlp over deprecated youtube_dl
except ImportError:
    yt_dlp = None

# Constants -------------------------------------------------------------------
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac', '.wma']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS
THEMES = ['Dark', 'Light', 'Blue', 'Green', 'Red', 'Purple', 'Professional', 'Midnight']
VISUALIZATION_MODES = ['Waveform', 'Spectrum', 'Spectrogram', 'Bars', 'Particles', 'Fire', 'Water']


# ============================================================================
# Audio Analyzer (thread, reusable, safe pause/resume)
# ============================================================================
class AudioAnalyzer(QThread):
    analysis_updated = pyqtSignal(object)

    def __init__(self, fft_size=2048, sample_rate=44100, history_size=20):
        super().__init__()
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.history_size = history_size
        self.window = np.hanning(fft_size)

        self._mutex = QMutex()
        self._wait = QWaitCondition()
        self._running = True           # thread lifetime
        self._paused = True            # data flow paused
        self._buffer = np.zeros(fft_size * 4, dtype=np.float32)
        self._buf_pos = 0

        self._freq_history = np.zeros((history_size, fft_size // 2), dtype=np.float32)
        self._spec_history = np.zeros((history_size, fft_size // 2), dtype=np.float32)
        self._hist_pos = 0

    # --- public API ---------------------------------------------------------
    def run(self):
        while self._running:
            self._mutex.lock()
            if self._paused or self._buf_pos < self.fft_size:
                self._wait.wait(self._mutex, 30)
                self._mutex.unlock()
                continue

            chunk = self._buffer[:self.fft_size].copy()
            # shift remaining samples to front
            remaining = self._buf_pos - self.fft_size
            if remaining > 0:
                self._buffer[:remaining] = self._buffer[self.fft_size:self._buf_pos]
            self._buf_pos = remaining
            self._mutex.unlock()

            windowed = chunk * self.window
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft) / self.fft_size
            power = 20.0 * np.log10(magnitude + 1e-12)

            self._freq_history[self._hist_pos] = magnitude[: self.fft_size // 2]
            self._spec_history[self._hist_pos] = power[: self.fft_size // 2]
            self._hist_pos = (self._hist_pos + 1) % self.history_size

            result = {
                'spectrum':    self._freq_history.mean(axis=0),
                'spectrogram': self._spec_history.mean(axis=0),
                'rms':         float(np.sqrt(np.mean(magnitude ** 2))),
                'peak':        float(magnitude.max()) if magnitude.size else 0.0,
            }
            self.analysis_updated.emit(result)

    def process_audio(self, buffer: QAudioBuffer):
        try:
            fmt = buffer.format()
            sample_fmt = fmt.sampleType()
            data = buffer.constData()
            if sample_fmt == fmt.Float:
                samples = np.frombuffer(bytes(data), dtype=np.float32)
            elif fmt.sampleSize() == 16:
                samples = np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(bytes(data), dtype=np.int32).astype(np.float32) / 2147483648.0
        except Exception:
            return

        with QMutexLocker(self._mutex):
            free = len(self._buffer) - self._buf_pos
            n = min(free, len(samples))
            if n > 0:
                self._buffer[self._buf_pos:self._buf_pos + n] = samples[:n]
                self._buf_pos += n
        self._wait.wakeAll()

    def resume(self):
        with QMutexLocker(self._mutex):
            self._paused = False
        self._wait.wakeAll()

    def pause(self):
        with QMutexLocker(self._mutex):
            self._paused = True

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False
        self._wait.wakeAll()
        self.wait(2000)


# ============================================================================
# Audio Visualizer Widget
# ============================================================================
class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mode = 'Spectrum'
        self._spectrum = np.zeros(512, dtype=np.float32)
        self._waveform = np.zeros(512, dtype=np.float32)
        self._phase = 0.0

    def set_visualization_mode(self, mode):
        self.mode = mode
        self.update()

    def update_visualizer(self, analysis):
        if not analysis:
            return
        spec = analysis.get('spectrum')
        if spec is not None:
            # resample to widget-friendly width
            target = 512
            if len(spec) >= target:
                idx = np.linspace(0, len(spec) - 1, target).astype(int)
                self._spectrum = spec[idx]
            else:
                self._spectrum = np.interp(
                    np.linspace(0, 1, target),
                    np.linspace(0, 1, len(spec)), spec)

        # synthesize a wavy waveform from RMS
        rms = analysis.get('rms', 0.0)
        amp = min(1.0, rms * 8.0)
        x = np.linspace(0, 1, 512)
        self._waveform = amp * np.sin(2 * np.pi * x * 8 + self._phase) * np.exp(-2 * x)
        self._phase += 0.4
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(20, 20, 25))

        if self.mode == 'Waveform':
            pen = QPen(QColor(80, 220, 160), 2)
            p.setPen(pen)
            mid = h / 2
            step = max(1, w // 512)
            for i in range(0, min(512, w)):
                y = mid + float(self._waveform[i]) * mid * 0.9
                p.drawPoint(i * step % w, int(y))
            # draw as polyline instead
            p.setPen(QPen(QColor(80, 220, 160), 2))
            pts = []
            for i in range(512):
                x = i * w / 512
                y = mid + float(self._waveform[i]) * mid * 0.9
                pts.append(QPoint(int(x), int(y)))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])

        elif self.mode in ('Spectrum', 'Bars'):
            p.setPen(Qt.NoPen)
            n = len(self._spectrum)
            bar_w = max(1, w // n)
            for i in range(n):
                v = min(1.0, float(self._spectrum[i]) * 6.0)
                bh = int(v * h)
                grad = QLinearGradient(0, h - bh, 0, h)
                grad.setColorAt(0.0, QColor(80, 220, 160))
                grad.setColorAt(1.0, QColor(30, 90, 200))
                p.setBrush(QBrush(grad))
                p.drawRect(i * bar_w, h - bh, bar_w - 1, bh)

        elif self.mode == 'Spectrogram':
            n = len(self._spectrum)
            cell_w = max(1, w // n)
            for i in range(n):
                v = min(1.0, float(self._spectrum[i]) * 4.0)
                color = QColor(int(255 * v), int(120 * v), int(255 * (1 - v)))
                p.fillRect(i * cell_w, 0, cell_w, h, color)

        else:  # Particles / Fire / Water — one simple generic mode
            p.setPen(Qt.NoPen)
            n = len(self._spectrum)
            for i in range(n):
                v = min(1.0, float(self._spectrum[i]) * 5.0)
                r = int(2 + v * 12)
                x = int(i * w / n)
                y = int(h - v * h * 0.9)
                p.setBrush(QColor(255, int(120 + 80 * v), int(40 * (1 - v)), 200))
                p.drawEllipse(QPoint(x, y), r, r)

        p.end()


# ============================================================================
# Lyrics Fetcher (cooperative cancel)
# ============================================================================
class LyricsFetcher(QThread):
    lyrics_fetched = pyqtSignal(str, str)

    def __init__(self, artist, title):
        super().__init__()
        self.artist = artist
        self.title = title
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        lyrics = None
        if requests is not None:
            try:
                lyrics = self._search_lyrics_ovh(self.artist, self.title)
                if not lyrics and not self._cancelled:
                    lyrics = self._search_lyrics_az(self.artist, self.title)
            except Exception as e:
                lyrics = f"Error fetching lyrics: {e}"
        else:
            lyrics = "Lyrics fetching requires `requests` and `beautifulsoup4`."
        if not self._cancelled:
            self.lyrics_fetched.emit(f"{self.artist} - {self.title}",
                                      lyrics or "Lyrics not found")

    def _search_lyrics_ovh(self, artist, title):
        try:
            url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                return r.json().get('lyrics', '').strip() or None
        except Exception:
            pass
        return None

    def _search_lyrics_az(self, artist, title):
        try:
            search_url = f"https://search.azlyrics.com/search.php?q={artist}+{title}"
            r = requests.get(search_url, timeout=8,
                             headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(r.text, 'html.parser')
            link = soup.find('a', href=lambda h: h and 'azlyrics.com/lyrics' in h)
            if not link:
                return None
            r2 = requests.get(link['href'], timeout=8,
                              headers={'User-Agent': 'Mozilla/5.0'})
            soup2 = BeautifulSoup(r2.text, 'html.parser')
            # azlyrics lyrics are inside a div with no class, between siblings
            divs = soup2.find_all('div', class_=False)
            for d in divs:
                txt = d.get_text("\n").strip()
                if len(txt) > 100:
                    return txt
        except Exception:
            pass
        return None


# ============================================================================
# Library Scanner (threaded so GUI doesn't freeze)
# ============================================================================
class LibraryScanner(QThread):
    file_found = pyqtSignal(str)
    finished_scan = pyqtSignal()

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for root, _, files in os.walk(self.directory):
            if self._cancelled:
                break
            for f in files:
                if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                    self.file_found.emit(os.path.join(root, f))
        self.finished_scan.emit()


# ============================================================================
# Media Library (metadata)
# ============================================================================
class MediaLibrary:
    def __init__(self):
        self.library = {}
        self.index = 0

    def add_file(self, path):
        fid = self.index
        self.index += 1
        md = self.get_metadata(path)
        self.library[fid] = {
            'id': fid, 'path': path,
            'title':  md.get('title') or os.path.basename(path),
            'artist': md.get('artist') or 'Unknown',
            'album':  md.get('album') or 'Unknown',
            'year':   md.get('year', ''),
            'genre':  md.get('genre', ''),
            'duration': md.get('duration', 0),
            'bitrate':  md.get('bitrate', 0),
            'lyrics':   md.get('lyrics', ''),
            'play_count': 0,
        }
        return fid

    def get_metadata(self, path):
        if not os.path.isfile(path):
            return {}
        try:
            if path.lower().endswith('.mp3') and eyed3 is not None:
                audio = eyed3.load(path)
                if audio and audio.tag and audio.info:
                    return {
                        'title':  audio.tag.title,
                        'artist': audio.tag.artist,
                        'album':  audio.tag.album,
                        'year':   str(audio.tag.getBestDate() or ''),
                        'genre':  str(audio.tag.genre or ''),
                        'duration': audio.info.time_secs,
                        'bitrate': audio.info.bit_rate[1] if audio.info.bit_rate else 0,
                        'lyrics': self._read_mp3_lyrics(audio),
                    }
            elif mutagen is not None:
                audio = mutagen.File(path, easy=True)
                info = mutagen.File(path)
                if audio is not None:
                    md = {
                        'title':  self._first(audio, 'title'),
                        'artist': self._first(audio, 'artist'),
                        'album':  self._first(audio, 'album'),
                        'year':   self._first(audio, 'date'),
                        'genre':  self._first(audio, 'genre'),
                    }
                    if info is not None and hasattr(info, 'info'):
                        md['duration'] = getattr(info.info, 'length', 0)
                        md['bitrate']  = getattr(info.info, 'bitrate', 0) // 1000
                    return md
        except Exception as e:
            print(f"[metadata] {path}: {e}")
        return {}

    @staticmethod
    def _first(audio, key):
        try:
            v = audio.get(key)
            return v[0] if v else None
        except Exception:
            return None

    @staticmethod
    def _read_mp3_lyrics(audio):
        try:
            for frame in audio.tag.frame_set.get('USLT', []):
                return frame.text
        except Exception:
            pass
        return ''


# ============================================================================
# Main Application
# ============================================================================
class UltimateMediaPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Media Player")
        self.resize(1200, 800)

        self.settings = QSettings("MediaPlayerCorp", "UltimateMediaPlayer")

        # ---- state ----------------------------------------------------------
        self.current_theme = "Dark"
        self.last_folder = QStandardPaths.writableLocation(QStandardPaths.MusicLocation)
        self.recent_files = []
        self.saved_volume = 50
        self.current_visualization = 'Spectrum'
        self.video_active = False

        # ---- media ----------------------------------------------------------
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setNotifyInterval(50)
        self.playlist = QMediaPlaylist()
        self.media_player.setPlaylist(self.playlist)

        self.video_widget = QVideoWidget()
        self.video_widget.setAspectRatioMode(Qt.KeepAspectRatio)

        self.visualizer = AudioVisualizer()
        self.visualizer.set_visualization_mode(self.current_visualization)

        # ---- audio analysis -------------------------------------------------
        self.audio_analyzer = AudioAnalyzer()
        self.audio_analyzer.analysis_updated.connect(self.visualizer.update_visualizer)
        self.audio_analyzer.start()

        self.audio_probe = QAudioProbe()
        self.audio_probe.setSource(self.media_player)
        self.audio_probe.audioBufferProbed.connect(self.audio_analyzer.process_audio)

        # ---- misc -----------------------------------------------------------
        self.media_library = MediaLibrary()
        self.lyrics_fetcher = None
        self.library_scanner = None

        self.sleep_timer = QTimer(self)
        self.sleep_timer.setSingleShot(True)
        self.sleep_timer.timeout.connect(self._sleep_timer_triggered)

        # ---- ui -------------------------------------------------------------
        self.load_settings()
        self._build_ui()
        self._connect_signals()
        self._init_shortcuts()
        self._init_tray()
        self.apply_theme(self.current_theme)
        self._restore_state()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Media / visualizer container
        self.media_container = QWidget()
        ml = QVBoxLayout(self.media_container)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.video_widget)
        ml.addWidget(self.visualizer)
        self.video_widget.hide()
        outer.addWidget(self.media_container, 1)

        # Track info
        info = QHBoxLayout()
        self.track_info_label = QLabel("No track loaded")
        self.track_info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.album_info_label = QLabel("")
        self.bitrate_label = QLabel("")
        self.media_type_label = QLabel("Audio")
        info.addWidget(self.track_info_label)
        info.addStretch(1)
        info.addWidget(self.album_info_label)
        info.addSpacing(20)
        info.addWidget(self.bitrate_label)
        info.addSpacing(20)
        info.addWidget(self.media_type_label)
        outer.addLayout(info)

        # Progress row
        prog = QHBoxLayout()
        self.current_time_label = QLabel("00:00:00")
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.total_time_label = QLabel("00:00:00")
        prog.addWidget(self.current_time_label)
        prog.addWidget(self.progress_slider, 1)
        prog.addWidget(self.total_time_label)
        outer.addLayout(prog)

        # Controls row
        ctrl = QHBoxLayout()
        self.shuffle_button = QPushButton("🔀")
        self.shuffle_button.setCheckable(True)
        self.prev_button = QPushButton("⏮")
        self.play_button = QPushButton("▶")
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.stop_button = QPushButton("⏹")
        self.next_button = QPushButton("⏭")
        self.repeat_button = QPushButton("🔁")
        self.repeat_button.setCheckable(True)
        self.visualizer_button = QPushButton("📊")
        self.visualizer_button.setCheckable(True)
        self.visualizer_button.setChecked(True)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(self.saved_volume)
        self.volume_slider.setFixedWidth(120)

        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems(VISUALIZATION_MODES)

        for w in (self.shuffle_button, self.prev_button, self.play_button,
                  self.stop_button, self.next_button, self.repeat_button,
                  self.visualizer_button):
            ctrl.addWidget(w)
        ctrl.addStretch(1)
        ctrl.addWidget(QLabel("Vol"))
        ctrl.addWidget(self.volume_slider)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Viz"))
        ctrl.addWidget(self.viz_mode_combo)
        outer.addLayout(ctrl)

        # Menus / toolbars / docks / status
        self._build_menus()
        self._build_toolbar()
        self._build_playlist_dock()
        self._build_lyrics_dock()
        self._build_library_dock()
        self._build_status_bar()

    def _build_menus(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        self._act(file_menu, "Open Files…", self.add_files, QKeySequence.Open)
        self._act(file_menu, "Open Folder…", self.open_folder)
        file_menu.addSeparator()
        self._act(file_menu, "Exit", self.close, QKeySequence.Quit)

        edit_menu = mb.addMenu("&Edit")
        self._act(edit_menu, "Clear Playlist", self.clear_playlist)

        view_menu = mb.addMenu("&View")
        self._act(view_menu, "Toggle Playlist", lambda: self.playlist_dock.setVisible(not self.playlist_dock.isVisible()))
        self._act(view_menu, "Toggle Lyrics",   lambda: self.lyrics_dock.setVisible(not self.lyrics_dock.isVisible()))
        self._act(view_menu, "Toggle Library",  lambda: self.library_dock.setVisible(not self.library_dock.isVisible()))
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for t in THEMES:
            a = QAction(t, self, checkable=True)
            a.setChecked(t == self.current_theme)
            a.triggered.connect(lambda _, name=t: self.apply_theme(name))
            theme_group.addAction(a)
            theme_menu.addAction(a)

        playback_menu = mb.addMenu("&Playback")
        self._act(playback_menu, "Play/Pause", self.play_pause, QKeySequence("Space"))
        self._act(playback_menu, "Stop", self.stop, QKeySequence("Ctrl+."))
        self._act(playback_menu, "Previous", self.previous_track)
        self._act(playback_menu, "Next", self.next_track)
        playback_menu.addSeparator()
        self._act(playback_menu, "Volume Up",   lambda: self.set_volume(min(100, self.media_player.volume() + 5)))
        self._act(playback_menu, "Volume Down", lambda: self.set_volume(max(0,   self.media_player.volume() - 5)))
        self._act(playback_menu, "Mute", self.toggle_mute)

        tools_menu = mb.addMenu("&Tools")
        self._act(tools_menu, "Fetch Lyrics for Current", self._fetch_lyrics_current)
        self._act(tools_menu, "Set Sleep Timer…", self._set_sleep_timer)

        help_menu = mb.addMenu("&Help")
        self._act(help_menu, "About", self._about)

    @staticmethod
    def _act(menu, text, slot, shortcut=None):
        a = QAction(text, menu)
        a.triggered.connect(slot)
        if shortcut is not None:
            a.setShortcut(shortcut)
        menu.addAction(a)
        return a

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction("Open", self.add_files)
        tb.addAction("Play/Pause", self.play_pause)
        tb.addAction("Stop", self.stop)
        tb.addAction("Previous", self.previous_track)
        tb.addAction("Next", self.next_track)

    def _build_playlist_dock(self):
        self.playlist_dock = QDockWidget("Playlist", self)
        self.playlist_widget = QListWidget()
        self.playlist_widget.itemDoubleClicked.connect(self._play_item)
        self.playlist_dock.setWidget(self.playlist_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.playlist_dock)

    def _build_lyrics_dock(self):
        self.lyrics_dock = QDockWidget("Lyrics", self)
        self.lyrics_widget = QTextEdit()
        self.lyrics_widget.setReadOnly(True)
        self.lyrics_dock.setWidget(self.lyrics_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.lyrics_dock)
        self.lyrics_dock.hide()

    def _build_library_dock(self):
        self.library_dock = QDockWidget("Media Library", self)
        self.library_widget = QTreeWidget()
        self.library_widget.setHeaderLabels(["Title", "Artist", "Album", "Duration"])
        self.library_widget.itemDoubleClicked.connect(self._play_library_item)
        self.library_dock.setWidget(self.library_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.library_dock)
        self.library_dock.hide()

    def _build_status_bar(self):
        self.playback_status_label = QLabel("Stopped")
        self.statusBar().addPermanentWidget(self.playback_status_label)
        self.statusBar().showMessage("Ready")

    def _init_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Left"),  self, lambda: self.seek_relative(-5000))
        QShortcut(QKeySequence("Ctrl+Right"), self, lambda: self.seek_relative(5000))
        QShortcut(QKeySequence("F11"),        self, self.toggle_fullscreen)
        QShortcut(QKeySequence("Esc"),        self, self.exit_fullscreen)
        QShortcut(QKeySequence("Ctrl+M"),     self, self.toggle_mute)

    def _init_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(self.windowIcon() or self.style().standardIcon(QStyle.SP_MediaPlay), self)
        menu = QMenu()
        menu.addAction("Show", self.showNormal)
        menu.addAction("Play/Pause", self.play_pause)
        menu.addAction("Quit", QApplication.instance().quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self.showNormal() if r == QSystemTrayIcon.DoubleClick else None)
        self.tray.show()

    # ---------------------------------------------------------------- signals
    def _connect_signals(self):
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.stateChanged.connect(self.update_playback_state)
        self.media_player.error.connect(self.handle_player_error)
        self.media_player.videoAvailableChanged.connect(self.video_availability_changed)
        self.media_player.volumeChanged.connect(self._on_volume_changed)

        self.playlist.currentIndexChanged.connect(self._playlist_index_changed)
        self.playlist.currentMediaChanged.connect(self.media_changed)

        self.progress_slider.sliderMoved.connect(self.set_position)
        self.volume_slider.valueChanged.connect(self.set_volume)

        self.play_button.clicked.connect(self.play_pause)
        self.stop_button.clicked.connect(self.stop)
        self.prev_button.clicked.connect(self.previous_track)
        self.next_button.clicked.connect(self.next_track)
        self.shuffle_button.toggled.connect(self.toggle_shuffle)
        self.repeat_button.toggled.connect(self.toggle_repeat)
        self.visualizer_button.toggled.connect(self.toggle_visualizer)
        self.viz_mode_combo.currentTextChanged.connect(self.set_visualization_mode)

    # ---------------------------------------------------------------- settings
    def load_settings(self):
        self.current_theme = self.settings.value("theme", "Dark")
        self.last_folder = self.settings.value(
            "lastFolder",
            QStandardPaths.writableLocation(QStandardPaths.MusicLocation))
        self.saved_volume = int(self.settings.value("volume", 50))
        self.recent_files = list(self.settings.value("recentFiles", []) or [])

    def save_settings(self):
        self.settings.setValue("theme", self.current_theme)
        self.settings.setValue("lastFolder", self.last_folder)
        self.settings.setValue("volume", self.media_player.volume())
        self.settings.setValue("recentFiles", self.recent_files)
        self.settings.setValue("maximized", self.isMaximized())
        self.settings.setValue("geometry", self.saveGeometry())

    def _restore_state(self):
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        if str(self.settings.value("maximized", "false")).lower() == "true":
            self.showMaximized()
        self.media_player.setVolume(self.saved_volume)

    # ---------------------------------------------------------------- actions
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Media Files", self.last_folder,
            "Media Files (*.mp3 *.wav *.ogg *.flac *.m4a *.aac *.wma *.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if files:
            self.last_folder = os.path.dirname(files[0])
            for f in files:
                self.add_to_playlist(f)
            if self.playlist.mediaCount() > 0 and self.media_player.state() != QMediaPlayer.PlayingState:
                self.playlist.setCurrentIndex(self.playlist.mediaCount() - len(files))
                self.media_player.play()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Scan Folder", self.last_folder)
        if not folder:
            return
        self.last_folder = folder
        self.library_widget.clear()
        if self.library_scanner and self.library_scanner.isRunning():
            self.library_scanner.cancel()
        self.library_scanner = LibraryScanner(folder)
        self.library_scanner.file_found.connect(self._library_file_found)
        self.library_scanner.finished_scan.connect(
            lambda: self.statusBar().showMessage("Library scan complete", 3000))
        self.library_scanner.start()
        self.library_dock.show()

    def _library_file_found(self, path):
        fid = self.media_library.add_file(path)
        md = self.media_library.library[fid]
        item = QTreeWidgetItem([
            md['title'], md['artist'], md['album'],
            self.format_time(int(md.get('duration', 0) * 1000)),
        ])
        item.setData(0, Qt.UserRole, path)
        self.library_widget.addTopLevelItem(item)

    def _play_library_item(self, item, _col):
        path = item.data(0, Qt.UserRole)
        if path:
            self.add_to_playlist(path)
            self.playlist.setCurrentIndex(self.playlist.mediaCount() - 1)
            self.media_player.play()

    def add_to_playlist(self, file_path):
        if not os.path.isfile(file_path):
            return
        media = QMediaContent(QUrl.fromLocalFile(file_path))
        self.playlist.addMedia(media)

        item = QListWidgetItem(os.path.basename(file_path))
        item.setData(Qt.UserRole, file_path)
        self.playlist_widget.addItem(item)

        if file_path not in self.recent_files:
            self.recent_files.insert(0, file_path)
            self.recent_files = self.recent_files[:10]

    def clear_playlist(self):
        self.playlist.clear()
        self.playlist_widget.clear()
        self.media_player.stop()

    def play_pause(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            if self.playlist.mediaCount() == 0:
                self.add_files()
                return
            if self.playlist.currentIndex() < 0:
                self.playlist.setCurrentIndex(0)
            self.media_player.play()

    def stop(self):
        self.media_player.stop()

    def previous_track(self):
        self.playlist.previous()

    def next_track(self):
        self.playlist.next()

    def set_volume(self, volume):
        self.media_player.setVolume(int(volume))
        if self.volume_slider.value() != int(volume):
            self.volume_slider.blockSignals(True)
            self.volume_slider.setValue(int(volume))
            self.volume_slider.blockSignals(False)

    def _on_volume_changed(self, v):
        if self.volume_slider.value() != v:
            self.volume_slider.blockSignals(True)
            self.volume_slider.setValue(v)
            self.volume_slider.blockSignals(False)

    def toggle_mute(self):
        if self.media_player.volume() == 0:
            self.set_volume(self.saved_volume or 50)
        else:
            self.saved_volume = self.media_player.volume()
            self.set_volume(0)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def seek_relative(self, ms):
        if self.media_player.duration() <= 0:
            return
        new_pos = self.media_player.position() + ms
        self.media_player.setPosition(max(0, min(new_pos, self.media_player.duration())))

    def toggle_shuffle(self, checked):
        self.playlist.setPlaybackMode(
            QMediaPlaylist.Random if checked else QMediaPlaylist.Sequential)

    def toggle_repeat(self, checked):
        self.playlist.setPlaybackMode(
            QMediaPlaylist.Loop if checked else QMediaPlaylist.Sequential)

    def toggle_visualizer(self, checked):
        if checked:
            self.video_widget.hide()
            self.visualizer.show()
            self.audio_analyzer.resume()
        else:
            self.visualizer.hide()
            if self.video_active:
                self.video_widget.show()
            self.audio_analyzer.pause()

    def set_visualization_mode(self, mode):
        self.current_visualization = mode
        self.visualizer.set_visualization_mode(mode)

    def update_position(self, position):
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def update_duration(self, duration):
        self.progress_slider.setRange(0, max(0, duration))
        self.total_time_label.setText(self.format_time(duration))

    def update_playback_state(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.playback_status_label.setText("Playing")
            if not self.video_active:
                self.audio_analyzer.resume()
        elif state == QMediaPlayer.PausedState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playback_status_label.setText("Paused")
            self.audio_analyzer.pause()
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playback_status_label.setText("Stopped")
            self.audio_analyzer.pause()

    def video_availability_changed(self, available):
        self.video_active = available
        if available:
            self.visualizer.hide()
            self.video_widget.show()
            self.visualizer_button.setChecked(False)
            self.audio_analyzer.pause()
            self.media_type_label.setText("Video")
        else:
            self.media_type_label.setText("Audio")
            if self.visualizer_button.isChecked():
                self.video_widget.hide()
                self.visualizer.show()
                self.audio_analyzer.resume()

    def _playlist_index_changed(self, index):
        if index >= 0 and index < self.playlist_widget.count():
            self.playlist_widget.setCurrentRow(index)

    def _play_item(self, item):
        row = self.playlist_widget.row(item)
        self.playlist.setCurrentIndex(row)
        self.media_player.play()

    def media_changed(self, media):
        if media.isNull():
            return
        url = media.canonicalUrl()
        if url.isLocalFile():
            path = url.toLocalFile()
            self._update_track_info(path)
            self._load_lyrics(path)
            self.statusBar().showMessage(f"Now playing: {os.path.basename(path)}", 5000)

    def _update_track_info(self, path):
        md = self.media_library.get_metadata(path)
        artist = md.get('artist') or 'Unknown'
        title = md.get('title') or os.path.basename(path)
        self.track_info_label.setText(f"{artist} — {title}")
        self.album_info_label.setText(md.get('album') or "")
        br = md.get('bitrate') or 0
        self.bitrate_label.setText(f"{br} kbps" if br else "")

    def _load_lyrics(self, path):
        md = self.media_library.get_metadata(path)
        embedded = md.get('lyrics') or ''
        if embedded:
            self.lyrics_widget.setPlainText(embedded)
            return
        artist = md.get('artist') or ''
        title = md.get('title') or os.path.splitext(os.path.basename(path))[0]
        if not (artist and title):
            self.lyrics_widget.setPlainText("Lyrics not available")
            return
        if self.lyrics_fetcher and self.lyrics_fetcher.isRunning():
            self.lyrics_fetcher.cancel()
            self.lyrics_fetcher.wait(500)
        self.lyrics_fetcher = LyricsFetcher(artist, title)
        self.lyrics_fetcher.lyrics_fetched.connect(self._update_lyrics)
        self.lyrics_fetcher.start()

    def _update_lyrics(self, _track, lyrics):
        self.lyrics_widget.setPlainText(lyrics)

    def _fetch_lyrics_current(self):
        url = self.media_player.currentMedia().canonicalUrl()
        if url.isLocalFile():
            self._load_lyrics(url.toLocalFile())

    def apply_theme(self, name):
        self.current_theme = name
        pal = QPalette()
        if name in ("Dark", "Midnight", "Professional"):
            pal.setColor(QPalette.Window, QColor(30, 30, 34) if name == "Midnight" else QColor(53, 53, 53))
            pal.setColor(QPalette.WindowText, Qt.white)
            pal.setColor(QPalette.Base, QColor(25, 25, 28))
            pal.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
            pal.setColor(QPalette.ToolTipBase, Qt.white)
            pal.setColor(QPalette.ToolTipText, Qt.white)
            pal.setColor(QPalette.Text, Qt.white)
            pal.setColor(QPalette.Button, QColor(53, 53, 53))
            pal.setColor(QPalette.ButtonText, Qt.white)
            pal.setColor(QPalette.BrightText, Qt.red)
            pal.setColor(QPalette.Link, QColor(90, 160, 230))
            pal.setColor(QPalette.Highlight, QColor(70, 130, 200))
            pal.setColor(QPalette.HighlightedText, Qt.black)
            QApplication.instance().setPalette(pal)
        elif name == "Light":
            QApplication.instance().setPalette(QApplication.style().standardPalette())
        elif name == "Blue":
            pal.setColor(QPalette.Window, QColor(28, 45, 80)); pal.setColor(QPalette.WindowText, Qt.white)
            QApplication.instance().setPalette(pal)
        elif name == "Green":
            pal.setColor(QPalette.Window, QColor(30, 60, 40)); pal.setColor(QPalette.WindowText, Qt.white)
            QApplication.instance().setPalette(pal)
        elif name == "Red":
            pal.setColor(QPalette.Window, QColor(70, 30, 30)); pal.setColor(QPalette.WindowText, Qt.white)
            QApplication.instance().setPalette(pal)
        elif name == "Purple":
            pal.setColor(QPalette.Window, QColor(55, 30, 70)); pal.setColor(QPalette.WindowText, Qt.white)
            QApplication.instance().setPalette(pal)
        self.settings.setValue("theme", name)

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()

    def _set_sleep_timer(self):
        minutes, ok = QInputDialog.getInt(self, "Sleep Timer", "Minutes:", 30, 1, 600)
        if ok:
            self.sleep_timer.start(minutes * 60 * 1000)
            self.statusBar().showMessage(f"Sleep timer set for {minutes} min", 3000)

    def _sleep_timer_triggered(self):
        self.media_player.pause()

    def _about(self):
        QMessageBox.about(self, "About",
                          "Ultimate Media Player\nVersion 2.1\n\nA PyQt5 demo player.")

    @staticmethod
    def format_time(ms):
        s = max(0, int(ms)) // 1000
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def handle_player_error(self):
        err = self.media_player.errorString()
        self.statusBar().showMessage(f"Playback error: {err}", 5000)

    # ---------------------------------------------------------------- shutdown
    def closeEvent(self, event):
        try:
            self.save_settings()
        except Exception:
            pass
        try:
            self.audio_analyzer.stop()
        except Exception:
            pass
        if self.lyrics_fetcher and self.lyrics_fetcher.isRunning():
            self.lyrics_fetcher.cancel()
            self.lyrics_fetcher.wait(1000)
        if self.library_scanner and self.library_scanner.isRunning():
            self.library_scanner.cancel()
            self.library_scanner.wait(1000)
        event.accept()


# ============================================================================
# Entry point
# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Ultimate Media Player")
    app.setApplicationVersion("2.1")
    app.setOrganizationName("MediaPlayerCorp")
    if platform.system() == "Windows":
        app.setStyle(QStyleFactory.create("Fusion"))

    window = UltimateMediaPlayer()
    window.show()

    def excepthook(t, v, tb):
        msg = "".join(traceback.format_exception(t, v, tb))
        print(msg, file=sys.stderr)
        try:
            QMessageBox.critical(window, "Error", f"{t.__name__}: {v}")
        except Exception:
            pass

    sys.excepthook = excepthook
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()