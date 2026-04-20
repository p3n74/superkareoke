import argparse
import sys
from PyQt6.QtWidgets import QApplication

from src.config import DB_PATH, APP_NAME
from src.logging_setup import configure_logging
from src.database.manager import DatabaseManager
from src.database.models import Song
from src.processing.pipeline import ProcessingPipeline
from src.ui.main_window import MainWindow
from src.ui.catalog_view import CatalogView
from src.ui.processing_view import ProcessingView
from src.ui.performance_view import PerformanceView
from src.ui.live_view import LiveView
from src.ui.settings_view import SettingsView


def main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose backend logs (includes more library noise).",
    )
    args, qt_argv = parser.parse_known_args()
    configure_logging(debug=args.debug)
    app = QApplication([sys.argv[0], *qt_argv])
    app.setApplicationName(APP_NAME)

    db = DatabaseManager(DB_PATH)

    window = MainWindow(db)

    # ── Create views ──
    catalog_view = CatalogView(db)
    processing_view = ProcessingView(db)
    settings_view = SettingsView()
    performance_view = PerformanceView(
        db,
        mic_device_fn=lambda: settings_view.selected_mic_index,
        output_device_fn=lambda: settings_view.selected_output_device,
        scoring_difficulty_fn=lambda: settings_view.scoring_difficulty_id,
        vocal_chain_settings_fn=lambda: settings_view.vocal_chain_settings,
        use_gpu_fn=lambda: settings_view.use_gpu,
    )
    live_view = LiveView()

    pipeline = ProcessingPipeline(
        db,
        use_gpu_fn=lambda: settings_view.use_gpu,
        processing_preset_fn=lambda: settings_view.processing_preset_id,
    )
    settings_view.gpu_preference_changed.connect(pipeline.reset_processors)
    settings_view.processing_preset_changed.connect(pipeline.reset_processors)
    settings_view.audio_devices_changed.connect(performance_view.apply_audio_devices)
    settings_view.vocal_chain_changed.connect(performance_view.apply_audio_devices)
    settings_view.load_settings_from_disk()
    performance_view.apply_audio_devices()

    window.set_page(0, catalog_view)
    window.set_page(1, processing_view)
    window.set_page(2, performance_view)
    window.set_page(3, live_view)
    window.set_page(4, settings_view)

    # ── Connect catalog → pipeline ──
    def on_process_requested(song: Song):
        song_from_db = db.get_song(song.id)
        if song_from_db is None:
            return
        processing_view.add_song(song_from_db)
        pipeline.process_song(song_from_db)
        window._switch_page(1)

    def on_sing_requested(song: Song):
        if song.id is None:
            return
        song_from_db = db.get_song(song.id)
        if song_from_db is None:
            return
        performance_view.load_song(song_from_db)
        window._switch_page(2)

    catalog_view.process_requested.connect(on_process_requested)
    catalog_view.sing_requested.connect(on_sing_requested)

    def on_song_deleted(song_id: int):
        processing_view.remove_song_card(song_id)
        performance_view.unload_song_if_removed(song_id)

    catalog_view.song_deleted.connect(on_song_deleted)
    app.aboutToQuit.connect(settings_view.save_settings_to_disk)

    # ── Connect pipeline → views ──
    pipeline.song_progress.connect(processing_view.update_progress)

    def on_song_finished(song_id: int, success: bool, error: str):
        processing_view.mark_finished(song_id, success, error)
        catalog_view.refresh()

    pipeline.song_finished.connect(on_song_finished)

    window.show()
    ret = app.exec()

    pipeline.shutdown()
    db.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
