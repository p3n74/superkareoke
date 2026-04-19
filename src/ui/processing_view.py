from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QScrollArea, QPushButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.database.manager import DatabaseManager
from src.database.models import Song, SongStatus


class SongProgressCard(QWidget):
    def __init__(self, song: Song, parent=None):
        super().__init__(parent)
        self.song = song
        self.setFixedHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)

        top = QHBoxLayout()
        self.title_label = QLabel(f"{song.title} — {song.artist}")
        self.title_label.setStyleSheet("color: #e0e0e0; font-size: 13px; font-weight: bold;")
        top.addWidget(self.title_label)
        top.addStretch()

        self.stage_label = QLabel("Queued")
        self.stage_label.setStyleSheet("color: #888; font-size: 11px;")
        top.addWidget(self.stage_label)
        layout.addLayout(top)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #0f3460; border: none; border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #ff5a75);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.setStyleSheet("""
            SongProgressCard {
                background: #16213e; border: 1px solid #0f3460; border-radius: 8px;
            }
        """)

    def update_progress(self, stage: str, progress: float):
        self.stage_label.setText(stage)
        self.progress_bar.setValue(int(progress * 100))

    def mark_finished(self, success: bool, error: str = ""):
        if success:
            self.stage_label.setText("Ready")
            self.stage_label.setStyleSheet("color: #5cb85c; font-size: 11px;")
            self.progress_bar.setValue(100)
        else:
            self.stage_label.setText(f"Error: {error[:60]}")
            self.stage_label.setStyleSheet("color: #d9534f; font-size: 11px;")


class ProcessingView(QWidget):
    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self._cards: dict[int, SongProgressCard] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Processing Queue")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #e94560;")
        layout.addWidget(title)

        desc = QLabel("Songs are downloaded, vocals are isolated, and pitch maps are extracted.")
        desc.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(desc)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch()

        scroll.setWidget(self._cards_container)
        layout.addWidget(scroll)

    def add_song(self, song: Song):
        if song.id in self._cards:
            return
        card = SongProgressCard(song)
        self._cards[song.id] = card
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)

    def update_progress(self, song_id: int, stage: str, progress: float):
        card = self._cards.get(song_id)
        if card:
            card.update_progress(stage, progress)

    def mark_finished(self, song_id: int, success: bool, error: str = ""):
        card = self._cards.get(song_id)
        if card:
            card.mark_finished(success, error)
