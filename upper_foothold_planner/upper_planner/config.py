"""Minimal configuration helpers for the independent Isaac Gym runtime."""


class AttrDict(dict):
    """Nested dictionary with attribute access, matching checkpoint semantics."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def from_nested(cls, value):
        if isinstance(value, dict):
            return cls({key: cls.from_nested(item) for key, item in value.items()})
        if isinstance(value, list):
            return [cls.from_nested(item) for item in value]
        return value
