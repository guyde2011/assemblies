from brain import Brain
from assemblies.assembly_fun import Assembly
from assemblies.utils import fire_many


class ReadRecursive:

    name = 'default'

    @staticmethod
    def read(assembly: Assembly, *, brain: Brain):
        fire_many(brain, assembly.parents, assembly.area)
        return brain.winners[assembly.area]
