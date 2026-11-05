"""Read-only audit receipts shared across pandas frame copies.

pandas deep-copies ``DataFrame.attrs`` on every derived frame. SICdb interval
and ventilation receipts hold per-interval lists and per-hour records, so
repeated deep copies dominated SICdb respiratory extraction. A frozen receipt
is returned as-is by ``copy``/``deepcopy`` and refuses in-place mutation; an
owner that needs a different receipt builds a new one.
"""
from __future__ import annotations


def _deny(self, *args, **kwargs):
    raise TypeError("audit receipt is read-only; build a new receipt instead")


class FrozenList(list):
    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = _deny
    sort = reverse = __iadd__ = __imul__ = _deny

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        # Rebuild through the constructor: the default list reduction would
        # replay items through append/extend, which are denied.
        return (FrozenList, (list(self),))


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (FrozenDict, (dict(self),))


def freeze(value):
    """Return a read-only copy of nested JSON-like lists and dicts."""
    if isinstance(value, (FrozenList, FrozenDict)):
        return value
    if isinstance(value, dict):
        return FrozenDict((key, freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return FrozenList(freeze(item) for item in value)
    return value


def freeze_sic_attrs(frame):
    """Freeze the frame's ``sicdb_*`` receipts in place and return the frame."""
    for key, value in list(frame.attrs.items()):
        if key.startswith("sicdb_"):
            frame.attrs[key] = freeze(value)
    return frame
