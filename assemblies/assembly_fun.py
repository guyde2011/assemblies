# TODO: rename this file
from __future__ import annotations  # TODO: remove this allover, we are using python 3
from .read_driver import ReadDriver  # TODO: It shouldn't depend on directory structure.
from utils.blueprints.recordable import Recordable
from utils.implicit_resolution import ImplicitResolution
from utils.bindable import Bindable
from brain.components import Stimulus, Area, UniquelyIdentifiable
from typing import Iterable, Union, Tuple, TYPE_CHECKING, Set, Optional, Dict
from itertools import product

if TYPE_CHECKING:  # TODO: this is not needed. It's better to always import them.
    from brain import Brain
    from brain.brain_recipe import BrainRecipe

# TODO: Document this type annotation. remember that future contributers may not know why it is necessary to use a string
Projectable = Union['Assembly', Stimulus]

# TODO: Look at parameters as well? (Yonatan, for associate)
bound_assembly_tuple = ImplicitResolution(lambda instance, name:
                                          Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'brain')


@Recordable(('merge', True), 'associate',
            resolution=ImplicitResolution(
                lambda instance, name: Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'recording'))
class AssemblyTuple(object):
    # TODO: this is documentation of the `__init__` method, not of the class
    # TODO 2: type hint return values, in all class methods (this is extremely useful here, because operators are defined)
    """
    Wraps a tuple of assemblies with syntactic sugar, such as >> (merge).

    :param assemblies: a tuple containing the assemblies
    """

    def __init__(self, *assemblies: Assembly):
        # TODO: add safeguard - assemblies is not empty / null
        # TODO 2: verify we don't get AssemblyTuple by mistake
        self.assemblies: Tuple[Assembly, ...] = assemblies

    # TODO: This is confusing, because I expect Assembly + Assembly = Assembly.
    #       There are other solutions. Even just AssemblyTuple(ass1, ass2) >> area is
    #       better, but I'm sure you can do better than that.
    # TODO 2: type hint of return value
    def __add__(self, other: AssemblyTuple):
        """
        In the context of AssemblyTuples, + creates a new AssemblyTuple containing the members
        of both parts.

        :param other: the other AssemblyTuple we add
        :returns: the new AssemblyTuple
        """
        # TODO: raise an exception, from an indicative exception class, instead of `assert`. Asserts are used only for debug/testing
        assert isinstance(other, AssemblyTuple), "Assemblies can be concatenated only to assemblies"
        return AssemblyTuple(*(self.assemblies + other.assemblies))

    @bound_assembly_tuple
    def merge(self, area: Area, *, brain: Brain = None):
        return Assembly._merge(self.assemblies, area, brain=brain)

    @bound_assembly_tuple
    def associate(self, other: AssemblyTuple, *, brain: Brain = None):
        return Assembly._associate(self.assemblies, other.assemblies, brain=brain)

    # TODO: rename other
    def __rshift__(self, other: Area):
        """
        In the context of assemblies, >> symbolizes merge.
        Example: (within a brain context) (a1+a2+a3)>>area

        :param other: the area we merge into
        :return: the new merged assembly
        """
        # TODO: assert -> raise exception
        assert isinstance(other, Area), "Assemblies must be merged onto an area"
        return self.merge(other)

    def __iter__(self):
        return iter(self.assemblies)


@Recordable(('project', True), ('reciprocal_project', True))
@Bindable('brain')
class Assembly(UniquelyIdentifiable, AssemblyTuple):
    # TODO: It makes no logical sense for Assembly to inherit AssemblyTuple.
    # TODO: instead, they can inherit from a mutual `AssemblyOperator` class that defines the operators they both support
    """
    A representation of an assembly of neurons that can be binded to a specific brain
    in which it appears. An assembly is defined primarily by its parents - the assemblies
    and/or stimuli that were fired to create it.
    This class implements basic operations on assemblies (project, reciprocal_project,
    merge and associate) by using a reader object, which interacts with the brain directly.

    # TODO: the following is documentation of init method and not of the class
    :param parents: the Assemblies and/or Stimuli that were used to create the assembly
    :param area: an Area where the Assembly "lives"
    :param appears_in: an iterable containing every BrainRecipe in which the assembly appears
    :param reader: name of a read driver pulled from assembly_readers. defaults to 'default'
    """
    # TODO: rename `appears_in` param
    def __init__(self, parents: Iterable[Projectable], area: Area,
                 appears_in: Iterable[BrainRecipe] = None, reader: str = 'default'):
        # We hash an assembly using its parents (sorted by id) and area
        # this way equivalent assemblies have the same id.
        UniquelyIdentifiable.__init__(self, uid=hash((area, *sorted(parents, key=hash))))
        AssemblyTuple.__init__(self, self)

        self.parents: Tuple[Projectable, ...] = tuple(parents)
        self.area: Area = area
        self.reader = ReadDriver(reader)
        self.appears_in: Set[BrainRecipe] = set(appears_in or [])
        for recipe in self.appears_in:
            recipe.append(self)

    # TODO: this name is not indicative. Perhaps change to something like to_representative_neuron_subset..
    # TODO: reader.read is _very_ confusing with Assembly.read. Rename reader.
    # TODO: return set
    def identify(self, preserve_brain=False, *, brain: Brain) -> Tuple[int, ...]:
        return self.reader.read(self, brain, preserve_brain=preserve_brain)

    # TODO: typing for area
    @staticmethod
    def read(area, *, brain: Brain):
        assemblies: Set[Assembly] = brain.recipe.area_assembly_mapping[area]
        overlap: Dict[Assembly, float] = {}
        for assembly in assemblies:
            # TODO: extract calculation to function with indicative name
            overlap[assembly] = len(set(area.winners) & set(assembly.identify(preserve_brain=True, brain=brain)))/area.k
        return max(overlap.keys(), key=lambda x: overlap[x])  # TODO: return None below some threshold

    # TODO: document
    # TODO 2: rename
    # TODO 3: what is the use case of hook and does it make logical sense? should it be part of the constructor?
    # TODO 4: there is no existing reader with `update_hook`. either make such reader and test the code using it, or remove all update_hook usages
    def _update_hook(self, *, brain: Brain):
        self.reader.update_hook(brain, self)


    # TODO: throughout bindable classes, users might error and give the brain parameter even if the object is binded.
    #       Is this a problem? can you help the user not make any mistakes?
    # TODO: add option to manually change the assemblies' recipes
    def project(self, area: Area, *, brain: Brain = None, iterations: Optional[int] = None) -> Assembly:
        """
        Projects an assembly into an area.

        :param brain: the brain in which the projection happens
        :param area: the area in which the new assembly is going to be created
        :returns: resulting projected assembly
        """
        # TODO: assert -> exception
        # TODO 2: more verification? area is not None, area is inside the brain
        # TODO 3: check any edge cases in the dependency between area and brain
        assert isinstance(area, Area), "Project target must be an Area"
        projected_assembly: Assembly = Assembly([self], area, appears_in=self.appears_in)
        if brain is not None:

            neurons = self.identify(brain=brain)

            brain.connectome.winners[self.area] = list(neurons)

            # TODO: is it only for better performance? it seems to affect correctness
            # Replace=True for better performance
            # TODO: *** WRONG LOGIC *** - add mapping area->area
            brain.next_round({self.area: [area]}, replace=True, iterations=iterations or brain.repeat)

            projected_assembly._update_hook(brain=brain)

        # TODO: calling `bind_like` manually is error-prone because someone can forget it. can you make a decorator or a more automated way to do it?
        projected_assembly.bind_like(self)
        return projected_assembly

    # TODO: rename other
    def __rshift__(self, other: Area):
        """
        In the context of assemblies, >> represents project.
        Example: a >> A (a is an assembly, A is an area)

        :param other: the area into which we project
        :returns: the new assembly that was created
        """
        # TODO: assert -> exception
        assert isinstance(other, Area), "Assembly must be projected onto an area"
        return self.project(other)

    def reciprocal_project(self, area: Area, *, brain: Brain = None) -> Assembly:
        """
        Reciprocally projects an assembly into an area,
        creating a projected assembly with strong bi-directional links to the current one.

        :param brain: the brain in which the projection occurs
        :param area: the area into which we project
        :returns: Resulting projected assembly
        """
        projected_assembly: Assembly = self.project(area, brain=brain)
        projected_assembly.project(self.area, brain=brain)
        self._update_hook(brain=brain)

        return projected_assembly

    @staticmethod
    def _merge(assemblies: Tuple[Assembly, ...], area: Area, *, brain: Brain = None) -> Assembly:
        """
        Creates a new assembly with all input assemblies as parents.
        Practically creates a new assembly with one-directional links from parents
        ONLY CALL AS: Assembly.merge(...), as the function is invariant under input order.

        :param brain: the brain in which the merge occurs
        :param assemblies: the parents of the new merged assembly
        :param area: the area into which we merge
        :returns: resulting merged assembly
        """
        # TODO: use exceptions from indicative exception classes, instead of assert
        assert len(assemblies) != 0, "tried to merge with empty input"
        # TODO: type hint is redundant
        # TODO 2: check documentation of `intersection` - it seems to be an instance method that works here by chance!
        merged_assembly: Assembly = Assembly(assemblies, area,
                                             appears_in=set.intersection(*[x.appears_in for x in assemblies]))
        # TODO: this is actually a way to check if we're in "binded" or "non binded" state.
        # TODO: can you think of a nicer way to do that?
        # TODO: otherwise it seems like a big block of code inside the function that sometimes happens and sometimes not. it is error-prone
        if brain is not None:
            # create a mapping from the areas to the neurons we want to fire
            area_neuron_mapping = {ass.area: [] for ass in assemblies}
            for ass in assemblies:
                # TODO: What happens if we merge assemblies that are already in the same area?
                area_neuron_mapping[ass.area] = list(ass.identify(brain=brain)) #maybe preserve brain?  # TODO: What does this comment mean?

            # update winners for relevant areas in the connectome
            for source in area_neuron_mapping.keys():
                brain.connectome.winners[source] = area_neuron_mapping[source]

            # Replace=True for better performance
            brain.next_round(subconnectome={source: [area] for source in area_neuron_mapping}, replace=True, iterations=brain.repeat)

            merged_assembly._update_hook(brain=brain)
        merged_assembly.bind_like(*assemblies)
        return merged_assembly

    @staticmethod
    def _associate(a: Tuple[Assembly, ...], b: Tuple[Assembly, ...], *, brain: Brain = None) -> None:
        # TODO: it's not the right logic
        """
        Associates two lists of assemblies, by strengthening each bond in the
        corresponding bipartite graph.
        for simple binary operation use Assembly.associate([a],[b]).
        for each x in A, y in B, associate (x,y).
        A1 z-z B1
        A2 -X- B2
        A3 z-z B3

        :param a: first list
        :param b: second list
        """
        pairs = product(a, b)
        for x, y in pairs:
            x.project(y.area, brain=brain)
            y.project(x.area, brain=brain)

    # TODO: lt and gt logic can be implemented using a common method
    def __lt__(self, other: Assembly):
        """
        Checks that other is a child assembly of self.

        :param other: the assembly we compare against
        """
        return isinstance(other, Assembly) and other in self.parents

    def __gt__(self, other: Assembly):
        """
        Checks if self is a child assembly of other.

        :param other: the assembly we compare against
        """
        return isinstance(other, Assembly) and self in other.parents
