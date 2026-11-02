"""Synthetic durable dispatcher shared by host tests."""


class Dispatcher:
    def __init__(self, path, callback, **kwargs):
        self.callback = callback
        self.submitted = []
        self.state = None

    def launch(self):
        pass

    def submit(self, identity):
        self.submitted.append(identity)
        return identity

    def status(self, identity):
        return self.state

    def close(self, timeout=4):
        return True

