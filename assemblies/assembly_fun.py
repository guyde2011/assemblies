from __future__ import annotations
from brain import Brain
from utils.implicit_resolution import ImplicitResolution
from utils.bindable import Bindable
from utils.repeat import Repeat
from brain.components import Stimulus, BrainPart, Area, UniquelyIdentifiable
from typing import Iterable, Union, Tuple, List, Dict, TYPE_CHECKING
from itertools import product

if TYPE_CHECKING:
    from brain.brain_recipe import BrainRecipe

Projectable = Union['Assembly', Stimulus]

bound_assembly_tuple = ImplicitResolution(lambda instance, name:
                                          Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'brain')
repeat = Repeat(resolve=lambda self, *args, **kwargs: self.t)
multiple_assembly_repeat = Repeat(resolve=lambda assembly1, assembly2, *args, **kwargs: max(assembly1.t, assembly2.t))

# TODO: Fix repeat (Yonatan)
repeat = lambda x: x
multiple_assembly_repeat = lambda x: x


# TODO: Eyal, add bindable to AssemblyTuple somehow, add more syntactic sugar

class AssemblyTuple(object):
    """
    Helper class for syntactic sugar, such as >> (merge)
    """

    def __init__(self, *assemblies: Assembly):
        self.assemblies: Tuple[Assembly] = assemblies

    def __add__(self, other: AssemblyTuple):
        """Add two assembly tuples"""
        assert isinstance(other, AssemblyTuple), "Assemblies can be concatenated only to assemblies"
        return Assembly.AssemblyTuple(self.assemblies + other.assemblies)

    @bound_assembly_tuple
    def __rshift__(self, other: Area, *, brain: Brain):
        """
        in the context of assemblies, >> symbolizes merge.
        Example: (within a brain context) (a1+a2+a3)>>area
        :param other:
        :param brain:
        :return:
        """
        assert isinstance(other, Area), "Assemblies must be merged onto an area"
        return Assembly.merge(self.assemblies, other)

    def __iter__(self):
        return iter(self.assemblies)


@Bindable('brain')
class Assembly(UniquelyIdentifiable, AssemblyTuple):
    """
    the main assembly object. according to our implementation, the main data of an assembly
    is his parents. we also keep a name for simulation puposes.
    TODO: Rewrite
    """

    def __init__(self, parents: Iterable[Projectable], area: Area, support_size: int, t: int = 1,
                 appears_in: Iterable[BrainRecipe] = None):
        """
        Initialize an assembly
        :param parents: Parents of the assembly (projectables that create it)
        :param area: Area the assembly "lives" in
        :param support_size: TODO: Tomer, fill in?
        :param t: Number of times to repeat each operation
        """

        UniquelyIdentifiable.__init__(self)
        AssemblyTuple.__init__(self, self)

        # Removed name from parameters
        self.parents: List[Projectable] = list(parents)
        self.area: Area = area
        self.support_size: int = support_size
        # TODO: Tomer, update to depend on brain as well?
        # Probably Dict[Brain, Dict[int, int]]
        self.support: Dict[int, int] = {}
        self.t: int = t
        self.appears_in: List[BrainRecipe] = list(appears_in or [])
        for recipe in self.appears_in:
            recipe.append(self)

    def _update_support(self, brain: Brain, winners: List[int]):
        # TODO: Tomer, need to index by brain
        # Something along the lines of self.supports[brain] = ...
        # TODO: Tomer, maybe add function for update_hook and then override it from the driver? Seems like the more "clean" way of approaching this

        oldest = 1
        for neuron in self.support:
            self.support[neuron] += 1
            oldest = max(oldest, self.support[neuron])
        for neuron in winners:
            self.support[neuron] = 1
        if len(self.support) <= self.support_size:
            return
        for neuron in self.support:
            if self.support[neuron] < oldest:
                continue
            del self.support[neuron]

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

    @repeat
    def project(self, area: Area, *, brain: Brain = None) -> 'Assembly':
        """
        Projects an assembly into an area
        :param brain:
        :param area:
        :return: Resulting projected assembly
        """
        assert isinstance(area, Area), "Project target must be an Area"
        projected_assembly: Assembly = Assembly([self], area, self.support_size, t=self.t, appears_in=self.appears_in)
        if brain is not None:
            Assembly.fire(brain, {self.area: [area]})
            # TODO: Tomer, update
            # projected_assembly.update_support(brain, brain.winners[area])

        projected_assembly.bind_like(self)
        return projected_assembly

    def __rrshift__(self, other: Area):
        assert isinstance(other, Area), "Assembly must be projected onto an area"
        self.project(other)

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
        Assembly.fire({projected_assembly: area})
        self.update_support(brain, brain.winners[self.area])
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

        # TODO: Which support size ot select? Maybe support size should be a global variable?
        # Lets think about this
        merged_assembly: Assembly = Assembly(assemblies, area, assemblies[0].support_size, t=assemblies[0].t,
                                             appears_in=set.intersection(*[x.appears_in for x in assemblies]))
        if brain is not None:
            Assembly.fire({ass: area for ass in assemblies})
            merged_assembly.update_support(brain, brain.winners[area])

        merged_assembly.bind_like(assemblies)
        return merged_assembly

    @staticmethod
    @multiple_assembly_repeat
    def _associate(a: [Assembly], b: [Assembly], *, brain: Brain = None) -> None:
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
        assert 0 not in [len(a), len(b)], "attempted to associate empty list"
        assert len(set([x.area for x in a + b])) <= 1, "can only associate assemblies in the same area"
        pairs = product(a, b)
        for x, y in pairs:
            Assembly.merge([x, y], x.area)

    def __lt__(self, other: 'Assembly'):
        """Checks that other is a child assembly of self"""
        return isinstance(other, Assembly) and other in self.parents

    def __gt__(self, other: 'Assembly'):
        """Checks if self is a child assembly of other"""
        return isinstance(other, Assembly) and self in other.parents
