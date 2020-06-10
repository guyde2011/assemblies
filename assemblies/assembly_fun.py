from __future__ import annotations
from .read_driver import ReadDriver
from brain import Brain
from utils.blueprints.recordable import Recordable
from utils.implicit_resolution import ImplicitResolution
from utils.bindable import Bindable
from utils.repeat import Repeat
from brain.components import Stimulus, BrainPart, Area, UniquelyIdentifiable
from typing import Iterable, Union, Tuple, List, Dict, TYPE_CHECKING, Set
from itertools import product

if TYPE_CHECKING:
    from brain.brain_recipe import BrainRecipe

Projectable = Union['Assembly', Stimulus]

bound_assembly_tuple = ImplicitResolution(lambda instance, name:
                                          Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'brain')
repeat = Repeat(resolve=lambda self, *args, **kwargs: self.t)
# TODO: Fix multiple_assembly_repeat (Eyal/Ido?)
multiple_assembly_repeat = Repeat(resolve=lambda assembly1, assembly2, *args, **kwargs: max(assembly1.t, assembly2.t))
multiple_assembly_repeat = lambda x: x


# TODO: Eyal, add bindable to AssemblyTuple somehow, add more syntactic sugar

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
        # TODO: Yonatan, Fix binding
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
    # TODO: Eyal, make this meaningfully uniquely identifiable (generate a hash on initialization and add an optional uid parameter in UniquelyIdentifiable)

    def __init__(self, parents: Iterable[Projectable], area: Area, t: int = 1,
                 appears_in: Iterable[BrainRecipe] = None, reader: str = 'default'):
        """
        Initialize an assembly
        :param parents: Parents of the assembly (projectables that create it)
        :param area: Area the assembly "lives" in
        :param t: Number of times to repeat each operation
        :param reader: Name of a read driver
        """

        UniquelyIdentifiable.__init__(self)
        AssemblyTuple.__init__(self, self)

        # Removed name from parameters
        self.parents: Tuple[Projectable, ...] = tuple(parents)
        self.area: Area = area
        # TODO: Tomer, update to depend on brain as well?
        # Probably Dict[Brain, Dict[int, int]]
        self.reader = ReadDriver(reader)
        self.t: int = t
        self.appears_in: Set[BrainRecipe] = set(appears_in or [])
        for recipe in self.appears_in:
            recipe.append(self)

    @staticmethod
    def _fire(mapping: Dict[BrainPart, List[BrainPart]], *, brain: Brain = None):
        """
        Fire an assembly
        TODO: Tomer, this is an helper function for you right? If so, move to where relevant
        TODO: Karidi, make sure brain binding here makes sense
        :param brain:
        :param mapping:
        :return:
        """
        # TODO: Ido & Eyal, this is supposed to be mapping into BrainPart not str, update if needed (?)
        # I updated the signature
        for p in mapping:
            for t in mapping[p]:
                # TODO: Ido & Eyal, handle the case the projectable is an assembly (?)
                brain.inhibit(p, t)
        brain.next_round()
        for p in mapping:
            for t in mapping[p]:
                brain.disinhibit(p, t)

    def read(self, *, brain: Brain) -> Tuple[int, ...]:
        return self.reader.read(self, brain)

    def update_hook(self, *, brain: Brain):
        self.reader.update_hook(self, brain)

    @repeat
    def project(self, area: Area, *, brain: Brain = None) -> 'Assembly':
        """
        Projects an assembly into an area
        :param brain:
        :param area:
        :return: Resulting projected assembly
        """
        assert isinstance(area, Area), "Project target must be an Area"
        projected_assembly: Assembly = Assembly([self], area, t=self.t, appears_in=self.appears_in)
        print("Firing", brain)
        if brain is not None:
            pass
            # Assembly.fire(brain, {self.area: [area]})
            # TODO: Tomer, update
            # projected_assembly.update_support(brain, brain.winners[area])

        projected_assembly.bind_like(self)
        return projected_assembly

    def __rshift__(self, other: Area):  # noqa
        assert isinstance(other, Area), "Assembly must be projected onto an area"
        return self.project(other)

    @repeat
    def reciprocal_project(self, area: Area, *, brain: Brain = None) -> 'Assembly':
        """
        Reciprocally projects an assembly into an area,
        creating a projected assembly with strong bi-directional links to the current one
        :param brain:
        :param area:
        :return: Resulting projected assembly
        """
        projected_assembly: Assembly = self.project(brain, area)
        # Assembly.fire({projected_assembly: area})
        return projected_assembly

    @staticmethod
    @multiple_assembly_repeat
    def _merge(assemblies: Tuple[Assembly, ...], area: Area, *, brain: Brain = None) -> 'Assembly':
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
        print("Merging...", brain)

        # Lets think about this
        merged_assembly: Assembly = Assembly(assemblies, area, t=assemblies[0].t,
                                             appears_in=set.intersection(*[x.appears_in for x in assemblies]))
        if brain is not None:
            pass
            # Assembly.fire({ass: area for ass in assemblies})
            #merged_assembly.update_support(brain, brain.winners[area])

        merged_assembly.bind_like(*assemblies)
        return merged_assembly

    @staticmethod
    @multiple_assembly_repeat
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
        #assert 0 not in [len(a), len(b)], "attempted to associate empty list"
        # Yonatan: Let's talk about this but maybe we allow associate also from different areas?
        # looks like a nice feature
        # assert len(set([x.area for x in a + b])) <= 1, "can only associate assemblies in the same area"
        print("Associating...", brain)
        pairs = product(a, b)
        for x, y in pairs:
            Assembly._merge((x, y), x.area, brain=brain)  # Eyal: You omitted brain, notice that you need to specify it

    def __lt__(self, other: 'Assembly'):
        """Checks that other is a child assembly of self"""
        return isinstance(other, Assembly) and other in self.parents

    def __gt__(self, other: 'Assembly'):
        """Checks if self is a child assembly of other"""
        return isinstance(other, Assembly) and self in other.parents
