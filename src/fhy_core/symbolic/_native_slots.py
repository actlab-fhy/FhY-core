"""Slot filling for public classes that subclass a Rust class.

A public class built on a Rust class declares slots named after the Rust
class's getters, so reading an attribute is a slot read rather than a call
into the extension. The slots are filled once, at construction.
"""

__all__ = ["copy_native_attributes"]


def copy_native_attributes(
    instance: object, public: type, native: type, *names: str
) -> None:
    """Copy the attributes `names` of `instance` from its Rust class into its slots.

    Called by ``__init__``, after the Rust class built the value: the slot
    descriptors of the public class shadow the Rust class's getters, so an
    attribute read afterwards is a slot read.
    """
    for name in names:
        getattr(public, name).__set__(instance, getattr(native, name).__get__(instance))
