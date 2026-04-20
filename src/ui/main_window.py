import sys

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon, QShortcut, QKeySequence

from src.config import APP_NAME, APP_VERSION
from src.database.manager import DatabaseManager


class SidebarButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        font = QFont()
        font.setPointSize(11)
        self.setFont(font)


class MainWindow(QMainWindow):
    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Sidebar ──
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 16, 8, 16)
        sidebar_layout.setSpacing(4)

        logo_label = QLabel(APP_NAME)
        logo_label.setObjectName("logoLabel")
        logo_font = QFont()
        logo_font.setPointSize(14)
        logo_font.setBold(True)
        logo_label.setFont(logo_font)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_label)
        sidebar_layout.addSpacing(24)

        self.nav_buttons: list[SidebarButton] = []
        nav_items = [
            ("Catalog", 0),
            ("Processing", 1),
            ("Sing", 2),
            ("Live Mode", 3),
            ("Settings", 4),
        ]
        for label, idx in nav_items:
            btn = SidebarButton(label)
            btn.clicked.connect(lambda checked, i=idx: self._switch_page(i))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        sidebar_layout.addStretch()

        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #888; font-size: 10px;")
        sidebar_layout.addWidget(version_label)

        root_layout.addWidget(sidebar)

        # ── Page stack ──
        self.page_stack = QStackedWidget()
        root_layout.addWidget(self.page_stack)

        self._pages: dict[int, QWidget] = {}
        self._apply_styles()
        self._setup_shortcuts()
        self._switch_page(0)

    def set_page(self, index: int, widget: QWidget):
        if index in self._pages:
            self.page_stack.removeWidget(self._pages[index])
        self.page_stack.insertWidget(index, widget)
        self._pages[index] = widget

    def _switch_page(self, index: int):
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        if index < self.page_stack.count():
            self.page_stack.setCurrentIndex(index)

    def _setup_shortcuts(self):
        for i in range(5):
            nav = i + 1
            if sys.platform == "darwin":
                seq = QKeySequence(f"Meta+{nav}")
            else:
                seq = QKeySequence(f"Ctrl+{nav}")
            shortcut = QShortcut(seq, self)
            shortcut.activated.connect(lambda idx=i: self._switch_page(idx))

        QShortcut(QKeySequence("F11"), self).activated.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }

            #sidebar {
                background: #16213e;
                border-right: 1px solid #0f3460;
            }
            #logoLabel { color: #e94560; }

            SidebarButton {
                background: transparent;
                color: #a8a8b3;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                text-align: left;
            }
            SidebarButton:hover { background: #1a2744; color: #ffffff; }
            SidebarButton:checked {
                background: #0f3460;
                color: #e94560;
                font-weight: bold;
            }

            QStackedWidget { background: #1a1a2e; }

            QLabel { color: #e0e0e0; }
        """)
