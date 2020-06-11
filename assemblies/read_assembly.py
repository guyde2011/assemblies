import sys
from pathlib import Path
import importlib
import re


class ReadAssembly:
    def __init__(self, reader_name):
        self.readers = {}
        assemblies = Path(__file__).parent.absolute()  # path to assemblies
        readers = assemblies / 'AssemblyReaders'
        sys.path.insert(0, str(assemblies))
        for path in readers.iterdir():
            if not (path.is_file() and path.suffix == '.py'):
                continue
            m = importlib.import_module(f'{readers.name}.{path.stem}')
            for name in m.__dict__:
                val = m.__dict__[name]
                if not (re.match('read*', name) and callable(val) and 'name' in val.__dict__.keys()):
                    continue
                self.readers[val.__dict__['name']] = val
        self.reader = self.readers[reader_name]

    def __call__(self, assembly):
        return self.reader(assembly)
