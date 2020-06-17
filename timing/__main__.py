import importlib
from pathlib import Path
import sys
import click

from . import timing, profiling

PROFILE = False


def run_timing(path: Path):
    for file in path.iterdir():
        if file.name.endswith('.py') and file.name.startswith('time_') and file.name != 'time_graphs.py':
            click.echo(f'\x1b[33m[INFO] Loading {file.name}\x1b[37m')
            module = importlib.import_module(f'{path.stem}.{file.name[:-3]}', path.stem)
            for member in module.__dict__:
                value = module.__dict__[member]
                to_run = value
                if not member.startswith('time_'):
                    continue
                click.echo(f'\x1b[33m[INFO] Running: \x1b[1m{member}\x1b[0m\x1b[37m')
                to_run = timing(to_run)
                if PROFILE and hasattr(value, '__profile__'):
                    to_run = profiling(to_run)
                to_run()


if __name__ == "__main__":
    arg = './timing'
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    run_timing(Path(arg))
