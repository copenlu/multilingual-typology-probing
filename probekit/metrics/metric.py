from typing import Any


class Metric:
    def __init__(self):
        pass

    def compute(self, *args, **kwargs) -> Any:
        return self._compute(*args, **kwargs)

    def _compute(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
