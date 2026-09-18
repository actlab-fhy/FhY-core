"""Hypothesis settings shared by the property test suites.

Every property runs the example count of the profile ``tests/conftest.py``
loads. A bare ``settings(max_examples=N)`` replaces that count rather than
capping it, so under a profile that runs fewer than ``N`` examples it runs
more. ``cap_max_examples`` only ever lowers the count.
"""

from hypothesis import settings

__all__ = ["cap_max_examples"]


def cap_max_examples(ceiling: int) -> settings:
    """Return the loaded profile's settings with at most ``ceiling`` examples.

    The returned settings run ``min(ceiling, profile.max_examples)``
    examples and take every other setting from the loaded profile, so they
    lower a profile count above ``ceiling`` and never raise one below it.
    The profile is read when this is called, which is at import time for a
    decorator or a state machine's ``TestCase.settings``;
    ``tests/conftest.py`` loads the profile before any test module is
    imported, in every ``pytest-xdist`` worker as well.

    Args:
        ceiling: Largest number of examples the settings may run.

    Returns:
        Settings derived from the loaded profile, with the capped example
        count.

    """
    profile = settings.get_profile(settings.get_current_profile_name())
    return settings(profile, max_examples=min(ceiling, profile.max_examples))
