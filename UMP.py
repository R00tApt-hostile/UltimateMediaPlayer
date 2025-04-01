import os
import sys
import json
import platform
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from PyQt5.QtCore import (Qt, QUrl, QTimer, QSize, QPoint, QRect, QSettings, 
                         QStandardPaths, QFileInfo, QByteArray, QBuffer, QIODevice,
                         QThread, pyqtSignal, QMutex, QLibraryInfo, QTranslator)
from PyQt5.QtGui import (QIcon, QPalette, QColor, QLinearGradient, QPainter, 
                         QBrush, QPixmap, QImage, QFont, QFontMetrics, QKeySequence,
                         QGuiApplication, QClipboard)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QSlider, QLabel, QPushButton, QFileDialog, QListWidget, 
                            QListWidgetItem, QComboBox, QStyle, QSizePolicy, QFrame,
                            QMessageBox, QMenu, QAction, QActionGroup, QSystemTrayIcon,
                            QProgressDialog, QShortcut, QScrollArea, QDockWidget, QToolBar,
                            QStatusBar, QInputDialog, QSpacerItem, QDialog, QDialogButtonBox,
                            QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                            QTabWidget, QTextEdit, QLineEdit, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtMultimedia import (QMediaPlayer, QMediaContent, QMediaPlaylist, 
                               QAudioProbe, QAudioBuffer, QAudio, QAudioEncoderSettings,
                               QMediaRecorder, QAudioRecorder, QAudioDeviceInfo)
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
import librosa
from PIL import Image, ImageFilter
import mutagen
from mutagen.id3 import ID3
import youtube_dl
import cdrom
import eyed3
import requests
from bs4 import BeautifulSoup

## Constants
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac', '.wma']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS
THEMES = ['Dark', 'Light', 'Blue', 'Green', 'Red', 'Purple', 'Professional', 'Midnight']
VISUALIZATION_MODES = ['Waveform', 'Spectrum', 'Spectrogram', 'Bars', 'Particles', 'Fire', 'Water']

## Utility Classes
class AudioAnalyzer(QThread):
    analysis_updated = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 44100
        self.fft_size = 2048
        self.history_size = 20
        self.freq_history = np.zeros((self.history_size, self.fft_size // 2))
        self.spec_history = np.zeros((self.history_size, self.fft_size // 2))
        self.current_pos = 0
        self.audio_buffer = np.zeros(self.fft_size * 2)
        self.buffer_pos = 0
        self.window = np.hanning(self.fft_size)
        self.mutex = QMutex()
        self.running = True
        
    def run(self):
        while self.running:
            self.mutex.lock()
            if self.buffer_pos >= self.fft_size:
                chunk = self.audio_buffer[:self.fft_size]
                self.audio_buffer = np.roll(self.audio_buffer, -self.fft_size)
                self.buffer_pos -= self.fft_size
                
                windowed = chunk * self.window
                fft = np.fft.rfft(windowed)
                magnitude = np.abs(fft) / self.fft_size
                power = 20 * np.log10(magnitude + 1e-12)
                
                self.freq_history[self.current_pos] = magnitude[:self.fft_size // 2]
                self.spec_history[self.current_pos] = power[:self.fft_size // 2]
                self.current_pos = (self.current_pos + 1) % self.history_size
                
                result = {
                    'spectrum': np.mean(self.freq_history, axis=0),
                    'spectrogram': np.mean(self.spec_history, axis=0),
                    'rms': np.sqrt(np.mean(np.square(magnitude))),
                    'peak': np.max(magnitude)
                }
                self.analysis_updated.emit(result)
            self.mutex.unlock()
            self.msleep(20)
            
    def process_audio(self, buffer):
        audio_data = np.frombuffer(buffer.data(), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        self.mutex.lock()
        available_space = len(self.audio_buffer) - self.buffer_pos
        if len(audio_data) <= available_space:
            self.audio_buffer[self.buffer_pos:self.buffer_pos+len(audio_data)] = audio_data
            self.buffer_pos += len(audio_data)
        self.mutex.unlock()
        
    def stop_analysis(self):
        self.running = False
        self.wait()

class CDRipper(QThread):
    progress_updated = pyqtSignal(int, str)
    rip_complete = pyqtSignal(str)
    
    def __init__(self, drive_path, output_format, output_dir):
        super().__init__()
        self.drive_path = drive_path
        self.output_format = output_format
        self.output_dir = output_dir
        self.cancel = False
        
    def run(self):
        try:
            # This would use platform-specific CD reading libraries
            # Simplified for example purposes
            tracks = 10  # Would actually detect from CD
            for i in range(1, tracks + 1):
                if self.cancel:
                    break
                
                self.progress_updated.emit(i, f"Ripping track {i}")
                # Simulate ripping
                self.msleep(1000)
                
                output_file = os.path.join(self.output_dir, f"track_{i}.{self.output_format}")
                # Actual ripping would happen here
                
                if not self.cancel:
                    self.rip_complete.emit(output_file)
                    
            self.progress_updated.emit(100, "Ripping complete" if not self.cancel else "Ripping cancelled")
        except Exception as e:
            self.progress_updated.emit(0, f"Error: {str(e)}")

class LyricsFetcher(QThread):
    lyrics_fetched = pyqtSignal(str, str)  # artist, lyrics
    
    def __init__(self, artist, title):
        super().__init__()
        self.artist = artist
        self.title = title
        
    def run(self):
        try:
            # Search various lyrics APIs
            lyrics = self.search_lyrics_ovh(self.artist, self.title)
            if not lyrics:
                lyrics = self.search_lyrics_az(self.artist, self.title)
                
            self.lyrics_fetched.emit(f"{self.artist} - {self.title}", lyrics or "Lyrics not found")
        except Exception as e:
            self.lyrics_fetched.emit(f"{self.artist} - {self.title}", f"Error fetching lyrics: {str(e)}")
            
    def search_lyrics_ovh(self, artist, title):
        try:
            url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json().get('lyrics', '')
        except:
            pass
        return None
        
    def search_lyrics_az(self, artist, title):
        try:
            search_url = f"https://search.azlyrics.com/search.php?q={artist}+{title}"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse results and get lyrics page
            # This would be more complex in reality
            lyrics_url = None
            for link in soup.find_all('a'):
                if "azlyrics.com/lyrics" in link.get('href', ''):
                    lyrics_url = link['href']
                    break
                    
            if lyrics_url:
                lyrics_response = requests.get(lyrics_url)
                lyrics_soup = BeautifulSoup(lyrics_response.text, 'html.parser')
                lyrics_div = lyrics_soup.find('div', class_=None)
                if lyrics_div:
                    return lyrics_div.get_text()
        except:
            pass
        return None

class MediaLibrary:
    def __init__(self):
        self.library = {}  # {id: {path, title, artist, album, year, genre, duration, ...}}
        self.index = 0
        
    def scan_directory(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                    path = os.path.join(root, file)
                    self.add_file(path)
                    
    def add_file(self, path):
        file_id = self.index
        self.index += 1
        
        metadata = self.get_metadata(path)
        self.library[file_id] = {
            'id': file_id,
            'path': path,
            'title': metadata.get('title', os.path.basename(path)),
            'artist': metadata.get('artist', 'Unknown'),
            'album': metadata.get('album', 'Unknown'),
            'year': metadata.get('year', ''),
            'genre': metadata.get('genre', ''),
            'duration': metadata.get('duration', 0),
            'bitrate': metadata.get('bitrate', 0),
            'last_played': None,
            'play_count': 0,
            'rating': 0,
            'lyrics': metadata.get('lyrics', ''),
            'tags': metadata.get('tags', [])
        }
        return file_id
        
    def get_metadata(self, path):
        try:
            if path.lower().endswith('.mp3'):
                audio = eyed3.load(path)
                if audio and audio.tag:
                    return {
                        'title': audio.tag.title,
                        'artist': audio.tag.artist,
                        'album': audio.tag.album,
                        'year': str(audio.tag.getBestDate()),
                        'genre': str(audio.tag.genre),
                        'duration': audio.info.time_secs,
                        'bitrate': audio.info.bit_rate[1],
                        'lyrics': self.get_lyrics(audio)
                    }
            else:
                # Handle other formats with mutagen
                audio = mutagen.File(path)
                if audio:
                    return {
                        'title': audio.get('title', [''])[0],
                        'artist': audio.get('artist', ['Unknown'])[0],
                        'album': audio.get('album', ['Unknown'])[0],
                        'year': audio.get('date', [''])[0],
                        'genre': audio.get('genre', [''])[0],
                        'duration': audio.info.length,
                        'bitrate': audio.info.bitrate
                    }
        except Exception as e:
            print(f"Error reading metadata: {str(e)}")
            
        return {}
        
    def get_lyrics(self, audiofile):
        try:
            for frame in audiofile.tag.frame_set:
                if frame.FrameID == 'USLT':
                    return frame.text
        except:
            pass
        return ''

## Main Application
class UltimateMediaPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize application
        self.setWindowTitle("Ultimate Media Player")
        self.setGeometry(100, 100, 1200, 800)
        
        # Application settings
        self.settings = QSettings("UltimateMediaPlayer", "MediaPlayer")
        self.load_settings()
        
        # Initialize components
        self.init_components()
        self.init_ui()
        self.init_shortcuts()
        self.init_system_tray()
        
        # Apply saved theme and settings
        self.apply_theme(self.current_theme)
        self.apply_settings()
        
        # Start background services
        self.start_services()
        
    def init_components(self):
        # Media components
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setNotifyInterval(50)
        self.playlist = QMediaPlaylist()
        self.media_player.setPlaylist(self.playlist)
        
        # Audio analysis
        self.audio_analyzer = AudioAnalyzer()
        self.audio_analyzer.analysis_updated.connect(self.update_visualizer)
        self.audio_probe = QAudioProbe()
        self.audio_probe.setSource(self.media_player)
        self.audio_probe.audioBufferProbed.connect(self.audio_analyzer.process_audio)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setAspectRatioMode(Qt.KeepAspectRatio)
        
        # Visualizer
        self.visualizer = AudioVisualizer()
        self.current_visualization = 'Spectrum'
        
        # Equalizer
        self.equalizer_bands = [0] * 10
        self.equalizer_enabled = False
        
        # Lyrics
        self.lyrics = {}
        self.current_lyric_index = -1
        self.lyrics_fetcher = None
        
        # Sleep timer
        self.sleep_timer = QTimer()
        self.sleep_timer.setSingleShot(True)
        self.sleep_timer.timeout.connect(self.sleep_timer_triggered)
        
        # Media library
        self.media_library = MediaLibrary()
        self.library_loaded = False
        
        # Recording
        self.audio_recorder = QAudioRecorder()
        self.setup_recorder()
        
        # CD ripping
        self.cd_ripper = None
        
        # Streaming
        self.youtube_dl_options = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(QStandardPaths.writableLocation(QStandardPaths.MusicLocation), 
                                   '%(title)s.%(ext)s')
        }
        
    def init_ui(self):
        # Main window setup
        self.setup_main_window()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_playlist()
        self.setup_controls()
        self.setup_status_bar()
        self.setup_dock_widgets()
        
        # Restore window state
        self.restore_state()
        
        # Connect signals
        self.connect_signals()
        
    def setup_main_window(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Media container
        self.media_container = QWidget()
        media_layout = QVBoxLayout()
        self.media_container.setLayout(media_layout)
        
        # Video and visualizer
        media_layout.addWidget(self.video_widget)
        media_layout.addWidget(self.visualizer)
        self.video_widget.hide()
        
        main_layout.addWidget(self.media_container, 1)
        
        # Progress controls
        self.setup_progress_controls(main_layout)
        
        # Playback controls
        self.setup_playback_controls(main_layout)
        
    def setup_menus(self):
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # Add all file actions...
        # (Previous implementation shown)
        
        # Edit menu
        edit_menu = self.menuBar().addMenu("&Edit")
        
        # Add edit actions...
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        # Add view actions...
        
        # Playback menu
        playback_menu = self.menuBar().addMenu("&Playback")
        
        # Add playback actions...
        
        # Tools menu
        tools_menu = self.menuBar().addMenu("&Tools")
        
        # Add tools actions...
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # Add help actions...

    def setup_toolbars(self):
        # Main toolbar
        self.main_toolbar = self.addToolBar("Main Toolbar")
        self.main_toolbar.setMovable(False)
        
        # Add toolbar actions...
        
        # Equalizer toolbar
        self.equalizer_toolbar = self.addToolBar("Equalizer")
        self.equalizer_toolbar.setVisible(False)
        
        # Add equalizer controls...

    def setup_playlist(self):
        # Playlist dock
        self.playlist_dock = QDockWidget("Playlist", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.playlist_dock)
        
        # Playlist widget
        self.playlist_widget = QListWidget()
        self.playlist_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.playlist_widget.customContextMenuRequested.connect(self.show_playlist_context_menu)
        
        self.playlist_dock.setWidget(self.playlist_widget)

    def setup_controls(self):
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Add control buttons...
        
        self.centralWidget().layout().addLayout(controls_layout)

    def setup_progress_controls(self, layout):
        # Progress slider and time labels
        progress_layout = QHBoxLayout()
        
        # Add progress controls...
        
        layout.addLayout(progress_layout)

    def setup_status_bar(self):
        # Status bar with playback info
        self.statusBar().showMessage("Ready")
        
        # Add status widgets...

    def setup_dock_widgets(self):
        # Lyrics dock
        self.lyrics_dock = QDockWidget("Lyrics", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.lyrics_dock)
        
        # Lyrics widget
        self.lyrics_widget = QLabel("No lyrics available")
        self.lyrics_dock.setWidget(self.lyrics_widget)
        self.lyrics_dock.hide()
        
        # Equalizer dock
        self.equalizer_dock = QDockWidget("Equalizer", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.equalizer_dock)
        
        # Equalizer widget
        equalizer_widget = QWidget()
        equalizer_layout = QVBoxLayout()
        
        # Add equalizer controls...
        
        equalizer_widget.setLayout(equalizer_layout)
        self.equalizer_dock.setWidget(equalizer_widget)
        self.equalizer_dock.hide()
        
        # Library dock
        self.library_dock = QDockWidget("Media Library", self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.library_dock)
        
        # Library widget
        self.library_widget = QTreeWidget()
        self.library_widget.setHeaderLabels(["Title", "Artist", "Album", "Duration"])
        self.library_dock.setWidget(self.library_widget)
        self.library_dock.hide()

    def connect_signals(self):
        # Media player signals
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.stateChanged.connect(self.update_playback_state)
        self.media_player.mediaStatusChanged.connect(self.update_media_status)
        self.media_player.error.connect(self.handle_player_error)
        self.media_player.videoAvailableChanged.connect(self.video_availability_changed)
        
        # Playlist signals
        self.playlist.currentIndexChanged.connect(self.playlist_index_changed)
        self.playlist.loaded.connect(self.playlist_loaded)
        self.playlist.currentMediaChanged.connect(self.media_changed)
        
        # Other signals
        self.playlist_widget.itemDoubleClicked.connect(self.play_selected_item)

    def start_services(self):
        # Start audio analyzer
        self.audio_analyzer.start()
        
        # Load media library in background
        self.load_media_library()
        
        # Check for CD drive
        self.check_cd_drive()

    def load_settings(self):
        # Load all settings...
        self.current_theme = self.settings.value("theme", "Dark")
        self.last_folder = self.settings.value("lastFolder", QStandardPaths.writableLocation(QStandardPaths.MusicLocation))
        
        # Load more settings...

    def save_settings(self):
        # Save all settings...
        self.settings.setValue("theme", self.current_theme)
        self.settings.setValue("lastFolder", self.last_folder)
        
        # Save more settings...

    def apply_settings(self):
        # Apply all settings...
        self.media_player.setVolume(int(self.settings.value("volume", 50)))
        
        # Apply more settings...

    def restore_state(self):
        # Restore window and dock states...
        if self.settings.value("maximized", False, type=bool):
            self.showMaximized()

    ## Core Functionality
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Media Files", self.last_folder,
            "Media Files (*.mp3 *.wav *.ogg *.flac *.mp4 *.avi *.mkv *.mov);;All Files (*)")
        
        if files:
            self.last_folder = os.path.dirname(files[0])
            for file in files:
                self.add_to_playlist(file)

    def add_to_playlist(self, file_path):
        # Add file to both playlist and playlist widget
        media = QMediaContent(QUrl.fromLocalFile(file_path))
        self.playlist.addMedia(media)
        
        item = QListWidgetItem(os.path.basename(file_path))
        item.setData(Qt.UserRole, file_path)
        self.playlist_widget.addItem(item)
        
        # Add to recent files
        if file_path not in self.recent_files:
            self.recent_files.append(file_path)
            if len(self.recent_files) > 10:
                self.recent_files.pop(0)

    def play_pause(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.audio_analyzer.stop_analysis()
        else:
            self.media_player.play()
            if not self.video_widget.isVisible():
                self.audio_analyzer.start()

    def stop(self):
        self.media_player.stop()
        self.audio_analyzer.stop_analysis()

    def previous_track(self):
        self.playlist.previous()

    def next_track(self):
        self.playlist.next()

    def set_volume(self, volume):
        self.media_player.setVolume(volume)
        self.settings.setValue("volume", volume)

    def toggle_mute(self):
        if self.media_player.volume() == 0:
            self.media_player.setVolume(self.saved_volume if hasattr(self, 'saved_volume') else 50)
        else:
            self.saved_volume = self.media_player.volume()
            self.media_player.setVolume(0)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def seek_relative(self, ms):
        new_pos = self.media_player.position() + ms
        self.media_player.setPosition(max(0, min(new_pos, self.media_player.duration())))

    def set_playback_rate(self, rate):
        self.media_player.setPlaybackRate(rate)

    def toggle_shuffle(self):
        if self.shuffle_button.isChecked():
            self.playlist.setPlaybackMode(QMediaPlaylist.Random)
        else:
            self.playlist.setPlaybackMode(QMediaPlaylist.Sequential)

    def toggle_repeat(self):
        if self.repeat_button.isChecked():
            self.playlist.setPlaybackMode(QMediaPlaylist.Loop)
        else:
            self.playlist.setPlaybackMode(QMediaPlaylist.Sequential)

    def toggle_visualizer(self):
        if self.visualizer_button.isChecked():
            self.video_widget.hide()
            self.visualizer.show()
            if self.media_player.state() == QMediaPlayer.PlayingState and not self.media_player.isVideoAvailable():
                self.audio_analyzer.start()
        else:
            self.video_widget.show()
            self.visualizer.hide()
            self.audio_analyzer.stop_analysis()

    def set_visualization_mode(self, mode):
        self.current_visualization = mode
        self.visualizer.set_visualization_mode(mode)

    def update_visualizer(self, analysis):
        self.visualizer.update_visualizer(analysis)

    def update_position(self, position):
        self.progress_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def update_duration(self, duration):
        self.progress_slider.setRange(0, duration)
        self.total_time_label.setText(self.format_time(duration))

    def update_playback_state(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.playback_status_label.setText("Playing")
            
            if not self.media_player.isVideoAvailable():
                self.audio_analyzer.start()
        elif state == QMediaPlayer.PausedState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playback_status_label.setText("Paused")
            self.audio_analyzer.stop_analysis()
        else:  # Stopped
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playback_status_label.setText("Stopped")
            self.audio_analyzer.stop_analysis()

    def video_availability_changed(self, available):
        if available:
            self.video_widget.show()
            self.visualizer.hide()
            self.audio_analyzer.stop_analysis()
            self.visualizer_button.setChecked(False)
            self.media_type_label.setText("Video")
        else:
            if self.visualizer_button.isChecked():
                self.video_widget.hide()
                self.visualizer.show()
                if self.media_player.state() == QMediaPlayer.PlayingState:
                    self.audio_analyzer.start()
            self.media_type_label.setText("Audio")

    def playlist_index_changed(self, index):
        if index >= 0:
            self.playlist_widget.setCurrentRow(index)

    def media_changed(self, media):
        current_url = media.canonicalUrl()
        if current_url.isLocalFile():
            file_path = current_url.toLocalFile()
            self.update_current_track_info(file_path)
            
            # Try to load lyrics
            self.load_lyrics(file_path)
            
            # Update status
            self.statusBar().showMessage(f"Now playing: {os.path.basename(file_path)}")

    def update_current_track_info(self, file_path):
        # Get metadata and update UI
        metadata = self.media_library.get_metadata(file_path)
        
        # Update info labels
        self.track_info_label.setText(f"{metadata.get('artist', 'Unknown')} - {metadata.get('title', os.path.basename(file_path))}")
        self.album_info_label.setText(metadata.get('album', 'Unknown'))
        
        # Update bitrate if available
        if 'bitrate' in metadata:
            self.bitrate_label.setText(f"{metadata['bitrate']} kbps")

    def load_lyrics(self, file_path):
        # Check for embedded lyrics first
        metadata = self.media_library.get_metadata(file_path)
        embedded_lyrics = metadata.get('lyrics', '')
        
        if embedded_lyrics:
            self.lyrics_widget.setText(embedded_lyrics)
            return
            
        # Try to fetch from internet
        artist = metadata.get('artist', '')
        title = metadata.get('title', os.path.splitext(os.path.basename(file_path))[0])
        
        if artist and title:
            if self.lyrics_fetcher:
                self.lyrics_fetcher.terminate()
                
            self.lyrics_fetcher = LyricsFetcher(artist, title)
            self.lyrics_fetcher.lyrics_fetched.connect(self.update_lyrics)
            self.lyrics_fetcher.start()

    def update_lyrics(self, track, lyrics):
        self.lyrics[track] = lyrics
        current_media = self.media_player.currentMedia().canonicalUrl()
        if current_media.isLocalFile():
            current_track = os.path.splitext(os.path.basename(current_media.toLocalFile()))[0]
            if track.startswith(current_track):
                self.lyrics_widget.setText(lyrics)

    def apply_theme(self, theme_name):
        # Apply theme to entire application
        self.current_theme = theme_name
        
        if theme_name == "Dark":
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            # Set all palette colors...
            self.setPalette(palette)
            
        elif theme_name == "Light":
            # Light theme palette...
            pass
            
        # More themes...
        
        self.settings.setValue("theme", theme_name)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()

    def format_time(self, milliseconds):
        seconds = milliseconds // 1000
        minutes = seconds // 60
        hours = minutes // 60
        return f"{hours:02d}:{minutes % 60:02d}:{seconds % 60:02d}"

    def handle_player_error(self):
        error = self.media_player.errorString()
        QMessageBox.warning(self, "Playback Error", f"An error occurred during playback:\n{error}")

    def closeEvent(self, event):
        # Clean up resources
        self.audio_analyzer.stop_analysis()
        self.audio_analyzer.quit()
        
        if self.lyrics_fetcher:
            self.lyrics_fetcher.terminate()
            
        if self.cd_ripper:
            self.cd_ripper.cancel = True
            self.cd_ripper.quit()
            
        # Save settings
        self.save_settings()
        
        event.accept()

## Entry Point
if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Ultimate Media Player")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("MediaPlayerCorp")
    
    # Set style
    if platform.system() == "Windows":
        app.setStyle("Fusion")
    
    # Create and show main window
    player = UltimateMediaPlayer()
    player.show()
    
    # Handle exceptions
    def excepthook(type, value, traceback):
        error_msg = f"Error: {str(value)}"
        QMessageBox.critical(player, "Error", error_msg)
        sys.__excepthook__(type, value, traceback)
    
    sys.excepthook = excepthook
    
    # Run application
    sys.exit(app.exec_())
