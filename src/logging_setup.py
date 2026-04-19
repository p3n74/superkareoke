"""Terminal logging for pipeline / backend diagnostics."""

import logging
import sys


def configure_logging(debug: bool = False) -> None:
    """Send logs to stderr. Use ``--debug`` for more detail and third-party loggers."""
    level = logging.DEBUG if debug else logging.INFO
    kwargs = dict(
        level=level,
        format="%(asctime)s [%(threadName)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if sys.version_info >= (3, 8):
        kwargs["force"] = True
    logging.basicConfig(**kwargs)
    if not debug:
        for name in ("urllib3", "torch", "numba", "matplotlib"):
            logging.getLogger(name).setLevel(logging.WARNING)
