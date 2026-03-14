from collections import OrderedDict


class BoundedTaskStore(OrderedDict):
    """
    OrderedDict with a fixed capacity.

    On overflow the oldest *completed* (done/failed) entry is evicted first.
    If every slot holds a pending/running task an OverflowError is raised so
    the caller can return HTTP 503 — active tasks are never silently dropped.
    """

    def __init__(self, maxsize: int = 200) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value) -> None:
        if key not in self and len(self) >= self._maxsize:
            evicted = next(
                (k for k, v in self.items() if v.status in ("done", "failed")),
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
