"""Self type support."""

__all__ = ["Self"]

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
