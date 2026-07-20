"""Thread-safe command and cancellation state for persistent model workers."""

import queue
import threading
from collections import deque


class WorkerControl:
    def __init__(self, max_pending_cancellations=1024):
        self.commands = queue.Queue()
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._current_request = None
        self._pending_cancellations = set()
        self._pending_order = deque()
        self._max_pending_cancellations = max(1, int(max_pending_cancellations))

    def submit(self, command):
        if command.get("action") != "cancel":
            self.commands.put(command)
            return
        request_id = command.get("request_id")
        with self._lock:
            if request_id not in self._pending_cancellations:
                self._pending_cancellations.add(request_id)
                self._pending_order.append(request_id)
            while len(self._pending_order) > self._max_pending_cancellations:
                self._pending_cancellations.discard(self._pending_order.popleft())
            if request_id == self._current_request:
                self.cancel_event.set()

    def begin(self, request_id):
        with self._lock:
            self._current_request = request_id
            if request_id in self._pending_cancellations:
                self.cancel_event.set()
            else:
                self.cancel_event.clear()

    def finish(self, request_id):
        with self._lock:
            if self._current_request == request_id:
                self._current_request = None
            self._pending_cancellations.discard(request_id)
            try:
                self._pending_order.remove(request_id)
            except ValueError:
                pass
            self.cancel_event.clear()

    @property
    def cancelled(self):
        return self.cancel_event.is_set()
