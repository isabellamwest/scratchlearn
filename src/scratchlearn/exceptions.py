"""Exceptions raised by scratchlearn."""


class NotFittedError(RuntimeError):
    """Raised when predict or transform is called before fit."""
