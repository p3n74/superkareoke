from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QDialog, QFormLayout, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from src.database.manager import DatabaseManager
from src.database.models import Song, SongStatus


STATUS_COLORS = {
    SongStatus.PENDING: "#888888",
    SongStatus.DOWNLOADING: "#f0ad4e",
    SongStatus.SEPARATING: "#5bc0de",
    SongStatus.EXTRACTING_PITCH: "#5bc0de",
    SongStatus.READY: "#5cb85c",
    SongStatus.ERROR: "#d9534f",
}


class AddSongDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Song Manually")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Song title")
        layout.addRow("Title:", self.title_edit)

        self.artist_edit = QLineEdit()
        self.artist_edit.setPlaceholderText("Artist name")
        layout.addRow("Artist:", self.artist_edit)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Add")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)

        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog { background: #1a1a2e; }
            QLabel { color: #e0e0e0; }
            QLineEdit {
                background: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
                border-radius: 4px; padding: 6px;
            }
            QPushButton {
                background: #0f3460; color: #e0e0e0; border: none;
                border-radius: 4px; padding: 8px 16px;
            }
            QPushButton:hover { background: #1a4a7a; }
        """)


_IN_PROGRESS = (
    SongStatus.DOWNLOADING,
    SongStatus.SEPARATING,
    SongStatus.EXTRACTING_PITCH,
)


class CatalogView(QWidget):
    song_selected = pyqtSignal(Song)
    process_requested = pyqtSignal(Song)
    sing_requested = pyqtSignal(Song)

    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("Song Catalog")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #e94560;")
        header.addWidget(title)
        header.addStretch()

        self.add_btn = QPushButton("+ Add Song")
        self.add_btn.setStyleSheet("""
            QPushButton {
                background: #e94560; color: white; border: none;
                border-radius: 6px; padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background: #ff5a75; }
        """)
        self.add_btn.clicked.connect(self._add_song)
        header.addWidget(self.add_btn)
        layout.addLayout(header)

        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search songs...")
        self.search_edit.setStyleSheet("""
            QLineEdit {
                background: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
                border-radius: 6px; padding: 10px; font-size: 13px;
            }
        """)
        self.search_edit.textChanged.connect(self._filter_table)
        layout.addWidget(self.search_edit)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Title", "Artist", "Status", "", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 130)
        self.table.setColumnWidth(3, 100)
        self.table.setColumnWidth(4, 100)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget {
                background: #16213e; color: #e0e0e0;
                border: 1px solid #0f3460; border-radius: 6px;
                gridline-color: #0f3460;
            }
            QTableWidget::item { padding: 8px; }
            QTableWidget::item:selected { background: #0f3460; }
            QHeaderView::section {
                background: #12192e; color: #888; border: none;
                padding: 8px; font-weight: bold;
            }
        """)
        layout.addWidget(self.table)

    def refresh(self):
        songs = self.db.get_all_songs()
        self._populate_table(songs)

    def _populate_table(self, songs: list[Song]):
        self.table.setRowCount(len(songs))
        for row, song in enumerate(songs):
            self.table.setItem(row, 0, QTableWidgetItem(song.title))
            self.table.setItem(row, 1, QTableWidgetItem(song.artist))

            status_item = QTableWidgetItem(song.status.value.replace("_", " ").title())
            color = STATUS_COLORS.get(song.status, "#888")
            status_item.setForeground(QColor(color))
            self.table.setItem(row, 2, status_item)

            # Process / Re-process button
            can_process = song.status not in _IN_PROGRESS
            proc_btn = QPushButton("Re-process" if song.status == SongStatus.READY else "Process")
            proc_btn.setEnabled(can_process)
            proc_btn.setToolTip(
                "Run or re-run download, separation, pitch, and lyrics."
                if song.status == SongStatus.READY
                else "Run the processing pipeline for this song."
            )
            if song.status in _IN_PROGRESS:
                proc_btn.setToolTip("Already processing — wait for it to finish or check the Processing page.")
            proc_btn.setStyleSheet("""
                QPushButton {
                    background: #0f3460; color: #ddd; border: none;
                    border-radius: 4px; padding: 4px 12px;
                }
                QPushButton:hover { background: #1a4a7a; }
                QPushButton:disabled { background: #333; color: #666; }
            """)
            proc_btn.clicked.connect(lambda _, s=song: self._on_process_clicked(s))
            self.table.setCellWidget(row, 3, proc_btn)

            # Sing button
            sing_btn = QPushButton("Sing!")
            sing_btn.setEnabled(song.status == SongStatus.READY)
            sing_btn.setStyleSheet("""
                QPushButton {
                    background: #e94560; color: white; border: none;
                    border-radius: 4px; padding: 4px 12px; font-weight: bold;
                }
                QPushButton:hover { background: #ff5a75; }
                QPushButton:disabled { background: #333; color: #666; }
            """)
            sing_btn.clicked.connect(lambda _, s=song: self.sing_requested.emit(s))
            self.table.setCellWidget(row, 4, sing_btn)

            self.table.setRowHeight(row, 44)

    def _filter_table(self, text: str):
        for row in range(self.table.rowCount()):
            title = self.table.item(row, 0).text().lower()
            artist = self.table.item(row, 1).text().lower()
            match = text.lower() in title or text.lower() in artist
            self.table.setRowHidden(row, not match)

    def _on_process_clicked(self, song: Song):
        if song.status == SongStatus.READY:
            r = QMessageBox.question(
                self,
                "Re-process song",
                f'Run the full pipeline again for "{song.title}"?\n\n'
                "This re-downloads from YouTube (when needed), replaces stems, pitch data, and synced lyrics. "
                "Your past performance scores stay in the database.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        self.process_requested.emit(song)

    def _add_song(self):
        dlg = AddSongDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            title = dlg.title_edit.text().strip()
            artist = dlg.artist_edit.text().strip()
            if not title:
                QMessageBox.warning(self, "Missing Title", "Please enter a song title.")
                return
            dup = self.db.find_duplicate_song(title, artist)
            if dup is not None:
                r = QMessageBox.question(
                    self,
                    "Possible duplicate",
                    f"A song with the same title and artist already exists:\n\n"
                    f'"{dup.title}" — {dup.artist or "(no artist)"}\n'
                    f"Status: {dup.status.value.replace('_', ' ').title()}\n\n"
                    "Add another entry anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if r != QMessageBox.StandardButton.Yes:
                    return
            song = Song(title=title, artist=artist)
            self.db.add_song(song)
            self.refresh()
