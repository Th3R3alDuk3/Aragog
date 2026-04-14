from collections import OrderedDict


class TaskStore(OrderedDict):
    def __init__(self, maxsize: int = 200) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value) -> None:
        if key not in self and len(self) >= self._maxsize:
            evicted = next(
                (item_key for item_key, item in self.items() if item.status in ("done", "failed")),
                None,
            )
            if evicted is not None:
                del self[evicted]
            else:
                raise OverflowError(
                    f"Task store is full ({self._maxsize}/{self._maxsize} active tasks). "
                    "Wait for running ingestions to complete or increase TASK_STORE_SIZE."
                )
        super().__setitem__(key, value)
