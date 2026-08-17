from app.logging_config import configure_logging


def test_configure_logging_is_idempotent() -> None:
    # Calling twice must not raise or double-register handlers.
    configure_logging()
    configure_logging()
