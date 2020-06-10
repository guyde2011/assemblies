import sys
from pathlib import Path
import importlib
import re


class ReadDriver:
    def __init__(self, reader_name):
        self.readers = {}
        assemblies = Path(__file__).parent.absolute()  # path to assemblies
        readers = assemblies / 'ReaderDrivers'
        sys.path.insert(0, str(assemblies))
        for path in readers.iterdir():
            if not (path.is_file() and path.suffix == '.py'):
                continue
            m = importlib.import_module(f'{readers.name}.{path.stem}')
            for name in m.__dict__:
                val = m.__dict__[name]
                if not (re.match('Read*', name) and 'name' in val.__dict__.keys()):
                    continue
                self.readers[val.__dict__['name']] = val
        self.reader = self.readers[reader_name]

    def read(self, brain: Brain):
        return self.reader.read(brain= brain)

    def update_hook(self, brain: Brain):
        if hasattr(self.reader, 'update_hook'):
            self.reader.update_hook(self, brain)
