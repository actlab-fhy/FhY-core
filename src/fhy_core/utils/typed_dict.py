"""Version-compatible ``ReadOnly`` and ``TypedDict``.

Re-exports :data:`typing.ReadOnly` and :class:`typing.TypedDict` on
Python 3.13+ and falls back to :mod:`typing_extensions` on earlier
versions, whose ``TypedDict`` records ``ReadOnly`` items in
``__readonly_keys__`` (PEP 705).

Note:
    Remove this module when Python 3.12 support is dropped. At that
    point, import ``ReadOnly`` and ``TypedDict`` directly from
    :mod:`typing` instead.

"""

import sys

if sys.version_info >= (3, 13):
    from typing import ReadOnly, TypedDict
else:
    from typing_extensions import ReadOnly, TypedDict

__all__ = ["ReadOnly", "TypedDict"]
