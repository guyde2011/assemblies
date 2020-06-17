import math
from typing import List, Union, Set
import numpy as np
from .components import *


class BlockConnection:
    def __init__(self, source, dest, blocks):
        self.source = source
        self.dest = dest
        self.blocks = blocks
        self.block_sizes = [len(block) for block in self.blocks]

    @property
    def beta(self):
        if isinstance(self.source, Stimulus):
            return self.dest.beta
        return self.source.beta

    def __getitem__(self, key):
        total = 0
        for idx, size in enumerate(self.block_sizes):
            if size >= key:
                return self.blocks[idx][key - total]
            total += size

    def __setitem__(self, key, value):
        total = 0
        for idx, size in enumerate(self.block_sizes):
            if size >= key:
                self.blocks[idx][key - total] = value
                return
            total += size

    def __iter__(self):
        return zip(self.block_sizes, self.blocks)

    def __repr__(self):
        return f"BlockConnection({self.synapses!r})"


__all__ = ['BlockConnection']
