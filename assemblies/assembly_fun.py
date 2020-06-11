from __future__ import annotations
from .read_driver import ReadDriver
from brain import Brain
from utils.blueprints.recordable import Recordable
from utils.implicit_resolution import ImplicitResolution
from utils.bindable import Bindable
from brain.components import Stimulus, Area, UniquelyIdentifiable
from typing import Iterable, Union, Tuple, TYPE_CHECKING, Set, Optional
from itertools import product

if TYPE_CHECKING:
    from brain.brain_recipe import BrainRecipe

Projectable = Union['Assembly', Stimulus]

# TODO: Look at parameters as well? (Yonatan, for associate)
bound_assembly_tuple = ImplicitResolution(lambda instance, name:
                                          Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'brain')


# TODO: Eyal, add more syntactic sugar

# write project (assuming read)

@Recordable(('merge', True), 'associate',
            resolution=ImplicitResolution(
                lambda instance, name: Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'recording'))
class AssemblyTuple(object):
    """
    Helper class for syntactic sugar, such as >> (merge)
    """

    def __init__(self, *assemblies: Assembly):
        self.assemblies: Tuple[Assembly, ...] = assemblies

    def __add__(self, other: AssemblyTuple):
        """Add two assembly tuples"""
        assert isinstance(other, AssemblyTuple), "Assemblies can be concatenated only to assemblies"
        return AssemblyTuple(*(self.assemblies + other.assemblies))

    @bound_assembly_tuple
    def merge(self, area: Area, *, brain: Brain = None):
        return Assembly._merge(self.assemblies, area, brain=brain)

    @bound_assembly_tuple
    def associate(self, other: AssemblyTuple, *, brain: Brain = None):
        return Assembly._associate(self.assemblies, other.assemblies, brain=brain)

    def __rshift__(self, other: Area):
        """
        in the context of assemblies, >> symbolizes merge.
        Example: (within a brain context) (a1+a2+a3)>>area
        :param other:
        :param brain:
        :return:
        """
        assert isinstance(other, Area), "Assemblies must be merged onto an area"
        return self.merge(other)

    def __iter__(self):
        return iter(self.assemblies)


@Recordable(('project', True), ('reciprocal_project', True))
@Bindable('brain')
class Assembly(UniquelyIdentifiable, AssemblyTuple):
    """
    the main assembly object. according to our implementation, the main data of an assembly
    is his parents. we also keep a name for simulation puposes.
    TODO: Rewrite
    """

    def __init__(self, parents: Iterable[Projectable], area: Area,
                 appears_in: Iterable[BrainRecipe] = None, reader: str = 'default'):
        """
        Initialize an assembly
        :param parents: Parents of the assembly (projectables that create it)
        :param area: Area the assembly "lives" in
        :param reader: Name of a read driver
        """

        # We hash an assembly using its parents (sorted by id) and area
        # this way equivalent assemblies have the same id.
        UniquelyIdentifiable.__init__(self, uid=hash((area, *sorted(parents, key=hash))))
        AssemblyTuple.__init__(self, self)

        self.parents: Tuple[Projectable, ...] = tuple(parents)
        self.area: Area = area
        # TODO: Tomer, update to depend on brain as well?
        # Probably Dict[Brain, Dict[int, int]]
        self.reader = ReadDriver(reader)
        self.appears_in: Set[BrainRecipe] = set(appears_in or [])
        for recipe in self.appears_in:
            recipe.append(self)

    def read(self, *, brain: Brain) -> Tuple[int, ...]:
        return self.reader.read(self, brain)

    def _update_hook(self, *, brain: Brain):
        # TODO: Tomer, fix
        # self.reader.update_hook(self, brain)
        pass

    def project(self, area: Area, *, brain: Brain = None, iterations: Optional[int] = None) -> Assembly:
        """
        Projects an assembly into an area
        :param brain:
        :param area:
        :param iterations:
        :return: Resulting projected assembly
        """
        assert isinstance(area, Area), "Project target must be an Area"
        projected_assembly: Assembly = Assembly([self], area, appears_in=self.appears_in)
        if brain is not None:
            neurons = self.read(brain=brain)

            # TODO: Eyal see my update to line after merge
            # LINE FOR AFTER MERGE WITH PERFORMANCE
            # brain.connectome.winners[self.area] = neurons

            # CURRENT TEMPORARY BOOTSTRAPPING LINE
            brain.connectome._winners[self.area] = set(neurons)

            brain.next_round({self.area: [area]}, replace=True, iterations=iterations or brain.repeat)

            projected_assembly._update_hook(brain=brain)

        projected_assembly.bind_like(self)
        return projected_assembly

    def __rshift__(self, other: Area):  # noqa
        assert isinstance(other, Area), "Assembly must be projected onto an area"
        return self.project(other)

    def reciprocal_project(self, area: Area, *, brain: Brain = None) -> Assembly:
        """
        Reciprocally projects an assembly into an area,
        creating a projected assembly with strong bi-directional links to the current one
        :param brain:
        :param area:
        :return: Resulting projected assembly
        """
        projected_assembly: Assembly = self.project(area, brain=brain)
        projected_assembly.project(self.area, brain=brain)
        self._update_hook(brain=brain)

        return projected_assembly

    @staticmethod
    def _merge(assemblies: Tuple[Assembly, ...], area: Area, *, brain: Brain = None) -> Assembly:
        """
        Creates a new assembly with all input assemblies as parents,
        practically creates a new assembly with one-directional links from parents
        ONLY CALL AS: Assembly.merge(...), as the function is invariant under input order.
        :param brain:
        :param assemblies:
        :param area:
        :return: Resulting merged assembly
        """
        assert len(assemblies) != 0, "tried to merge with empty input"

        # Lets think about this
        merged_assembly: Assembly = Assembly(assemblies, area,
                                             appears_in=set.intersection(*[x.appears_in for x in assemblies]))
        if brain is not None:
            # TODO: Eyal, do merge
            pass
            # Assembly.fire({ass: area for ass in assemblies})
            merged_assembly._update_hook(brain=brain)

        merged_assembly.bind_like(*assemblies)
        return merged_assembly

    @staticmethod
    def _associate(a: Tuple[Assembly, ...], b: Tuple[Assembly, ...], *, brain: Brain = None) -> None:
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
        # Yonatan: It is OK to associate empty lists?
        # assert 0 not in [len(a), len(b)], "attempted to associate empty list"
        # Yonatan: Let's talk about this but maybe we allow associate also from different areas?
        # looks like a nice feature
        # assert len(set([x.area for x in a + b])) <= 1, "can only associate assemblies in the same area"
        pairs = product(a, b)
        for x, y in pairs:
            Assembly._merge((x, y), x.area, brain=brain)  # Eyal: You omitted brain, notice that you need to specify it

    def __lt__(self, other: Assembly):
        """Checks that other is a child assembly of self"""
        return isinstance(other, Assembly) and other in self.parents

    def __gt__(self, other: Assembly):
        """Checks if self is a child assembly of other"""
        return isinstance(other, Assembly) and self in other.parents
