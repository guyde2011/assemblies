import sys
from pathlib import Path
import importlib
import re

from brain import Brain


# TODO: Add documentation!


class ReadDriver:
    def __init__(self, reader_name):
        self.readers = {}
        # Path to assemblies package
        assemblies_package_path = Path(__file__).parent.absolute()
        readers = assemblies_package_path / 'assembly_readers'
        sys.path.insert(0, str(assemblies_package_path))
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

    def read(self, assembly, brain: Brain):
        return self.reader.read(assembly, brain=brain)

    def update_hook(self, brain: Brain):
        # TODO: Tomer, what about the assembly that was updated? we need that as a parameter as well
        # Please fix
        if hasattr(self.reader, 'update_hook'):
            self.reader.update_hook(self, brain)
