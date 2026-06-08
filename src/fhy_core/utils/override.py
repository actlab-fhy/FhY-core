"""Version-compatible ``@override`` decorator.

Re-exports :func:`typing.override` on Python 3.12+ and falls back to
:func:`typing_extensions.override` on earlier versions.

Note:
    Remove this module when Python 3.11 support is dropped (i.e. when the
    minimum supported version becomes 3.12). At that point, import
    ``override`` directly from :mod:`typing` instead.

"""

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

__all__ = ["override"]
