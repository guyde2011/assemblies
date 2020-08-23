import sys
from pathlib import Path


class Logger:
    class StreamSplitter:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for stream in self.streams:
                stream.write(s)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    def __init__(self, path: Path):
        self.stdout = sys.stdout
        self.stdout_log = open(str(path) + '.txt', 'w')
        self.stderr = sys.stderr
        self.stderr_log = open(str(path) + '.err.txt', 'w')

    def __enter__(self):
        sys.stdout = Logger.StreamSplitter(self.stdout, self.stdout_log)
        sys.stderr = Logger.StreamSplitter(self.stderr, self.stderr_log)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
