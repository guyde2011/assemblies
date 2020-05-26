from typing import List, Union

from assemblies.assembly_fun import Assembly
from brain.components import Area, Stimulus, BrainPart


class BrainRecipe:
    def __init__(self, *parts: Union[BrainPart, Assembly]):
        self.areas: List[Area] = []
        self.stimuli: List[Stimulus] = []
        self.assemblies: List[Assembly] = []
        self.extend(*parts)

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
                assembly.appears_in.append(self)

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
