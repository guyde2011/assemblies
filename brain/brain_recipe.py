from __future__ import annotations
from typing import List, Union, TYPE_CHECKING, Dict, Optional

from assemblies.assembly_fun import Assembly
from brain.components import Area, Stimulus, BrainPart
from utils.blueprints.recording import Recording
if TYPE_CHECKING:
    from brain import Brain


class BrainRecipe:
    def __init__(self, *parts: Union[BrainPart, Assembly]):
        self.areas: List[Area] = []
        self.stimuli: List[Stimulus] = []
        self.assemblies: List[Assembly] = []
        self.extend(*parts)
        self.initialization: Recording = Recording()
        self.ctx_stack: List[Dict[Assembly, Recording]] = []

    def _add_area(self, area: Area):
        if area not in self.areas:
            self.areas.append(area)

    def _add_stimulus(self, stimulus: Stimulus):
        if stimulus not in self.stimuli:
            self.stimuli.append(stimulus)

    def _add_assembly(self, assembly: Assembly):
        if assembly not in self.assemblies:
            self.assemblies.append(assembly)
            if self not in assembly.appears_in:
                assembly.appears_in.add(self)

    def append(self, part: Union[Assembly, BrainPart]):
        if isinstance(part, Area):
            self._add_area(part)
        elif isinstance(part, Stimulus):
            self._add_stimulus(part)
        elif isinstance(part, Assembly):
            self._add_assembly(part)

    def extend(self, *parts: Union[Assembly, BrainPart]):
        for part in parts:
            self.append(part)

    def initialize(self, brain: Brain):
        self.initialization.play(brain=brain)

    def __enter__(self):
        current_ctx_stack: Dict[Assembly, Optional[Recording]] = {}

        for assembly in self.assemblies:
            if 'recording' in assembly.bound_params:
                current_ctx_stack[assembly] = assembly.bound_params['recording']
            assembly.bind(recording=self.initialization)

        self.ctx_stack.append(current_ctx_stack)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_ctx_stack: Dict[Assembly, Optional[Recording]] = self.ctx_stack.pop()

        for assembly in self.assemblies:
            assembly.unbind('recording')
            if assembly in current_ctx_stack:
                assembly.bind(recording=current_ctx_stack[assembly])
