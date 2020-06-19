from brain import Brain
from assemblies.assembly_fun import Assembly
from assemblies.utils import fire_many, revert_changes


# TODO: Add documentation!


class ReadRecursive:

    name = 'default'

    @staticmethod
    def read(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain):
        """

        :param assembly: Assembly
        :param preserve_brain: Do we want to change the brain state or not
        :param brain:
        :return:
        """
        changed_areas = fire_many(brain, assembly.parents, assembly.area, preserve_brain)
        read_value = brain.winners[assembly.area]
        if preserve_brain:
            revert_changes(brain, changed_areas)
        return read_value
