from typing import Tuple, NamedTuple, Set, Iterable

import logging
from typing import List, Dict
import numpy as np
from collections import defaultdict

from numpy import ndarray
from scipy.stats import binom, truncnorm
import math
import random

from ..components import *
from .connectome import Connectome


class LazyConnectome(Connectome):
    def __init__(self, p: float, areas=None, stimuli=None):
        super(LazyConnectome, self).__init__(p, areas, stimuli)
        self._initialize_parts((areas or []) + (stimuli or []))

    def subconnectome(self, connections: Dict[BrainPart, Area]) -> Connectome:
        pass

    def area_connections(self, area: Area) -> List[Area]:
        pass

    def add_area(self, area: Area):
        super().add_area(area)
        self._initialize_parts([area])

    def add_stimulus(self, stimulus: Stimulus):
        super().add_stimulus(stimulus)
        self._initialize_parts([stimulus])

    def _initialize_parts(self, parts: List[BrainPart]):
        """
        Initialize all the connections to and from the given brain parts.
        :param parts: List of stimuli and areas to initialize
        :return:
        """
        for part in parts:
            for other in self.areas + self.stimuli:
                self.connections[part, other] = Connection(part, other)
                if isinstance(part, Area) and part != other:
                    self.connections[other, part] = Connection(other, part)

    def _calc_prev_inputs(self, area: Area, sources: List[BrainPart]) -> List[float]:
        """
        Calculates the overall inputs to each neuron in the support of the area
        :param area: the area we project into
        :type area: Area
        :param sources: the list of brain parts connected to the area
        :type sources: List[BrainPart]
        :return: a list of floats
        """
        prev_inputs: List[float] = [0.] * area.support_size

        for src in sources:
            for i in range(area.support_size):
                prev_inputs[i] = sum((inp[i] for inp in self.connections[src, area].synapses if len(inp) >= i))

        return prev_inputs

    @staticmethod
    def _calculate_input_sizes(sources: List[BrainPart]) -> Dict[BrainPart, int]:
        """
        Calculates a mapping between the given sources and their input size
        :param sources: the list of brain parts connected to the area
        :type sources: List[BrainPart]
        :return: a dict mapping between a brain part and its input size
        """
        input_sizes: Dict[BrainPart, int] = {src: (src.k if isinstance(src, Area) else src.n) for src in sources}
        logging.debug(f'total_k = {sum(input_sizes.values())} and input_sizes = {input_sizes}')
        return input_sizes

    def _calc_potential_winners(self, area: Area, total_k: int) -> List[float]:
        # effective_n := Number of neurons that never fired in the area
        effective_n = area.n - area.support_size
        # Threshold for inputs that are above (n-k)/n percentile. alpha is the smallest number such that:
        # Pr(Bin(total_k,self.p) <= alpha) >= (effective_n-area.k)/effective_n
        # A.k.a the probability that the number of neurons that aren't going to fire in the area will be lower than
        # p * (number of neurons in the area that never fired)
        alpha = binom.ppf((float(effective_n - area.k) / effective_n), total_k, self.p)
        logging.debug(f'Alpha = {alpha}')
        # Std(Binomial(n,p)) := Sqrt(n * p * (1-p))
        std = math.sqrt(total_k * self.p * (1.0 - self.p))
        mu = total_k * self.p
        a = float(alpha - mu) / std
        b = float(total_k - mu) / std  # note that b>=a and corresponds to the maximum value of Bin(total_k,self.p)
        # We take effective_n samples from the distribution of the neuron inputs
        potential_winners = truncnorm.rvs(a, b, scale=std, loc=mu, size=effective_n)
        for i in range(area.k):
            potential_winners[i] = float(round(potential_winners[i]))
        logging.debug(f'Potential Winners: {potential_winners}')
        return list(potential_winners)

    @staticmethod
    def _calc_new_winners(area: Area, prev_winner_inputs: List[float], potential_new_winners: List[float]) \
            -> Tuple[List[float], ndarray]:
        """
        find area.k maximal values in prev_winner_inputs + potential_new_winners - these are the new winners.
        find the ones that ar e winners for the first time.
        :param area:
        :param prev_winner_inputs:
        :param potential_new_winners:
        :return: list of inputs of the new winners that weren't winners before (represented as a list of floats)
                 and the new winners
        """
        both = np.array(prev_winner_inputs + potential_new_winners)
        new_winners = np.argpartition(both, area.k)[:area.k]
        num_first_winners = 0
        first_winner_inputs = []
        for i in range(area.k):
            if new_winners[i] >= area.support_size:  # winner for the first time
                first_winner_inputs.append(both[new_winners[i]])
                new_winners[i] = area.support_size + num_first_winners
                num_first_winners += 1
        print(f"New Winners: {len(new_winners)}")
        return first_winner_inputs, new_winners

    def _project_into(self, area: Area, sources: List[BrainPart]) \
            -> Tuple[Dict[BrainPart, int], List[float], ndarray, int]:
        """Project multiple stimuli and area assemblies into area 'area' at the same time.
        :param area: The area projected into
        :param sources: List of separate brain parts whose assemblies we will projected into this area
        :return: input_sizes, first_winner_inputs, new_winners, new_support_size
        """
        # TODO Handle case of projecting from an area without previous winners.
        # TODO: there is a bug when adding a new stimulus later on.
        # TODO: Stimulus is updating to somehow represent >100 neurons.
        logging.info(f'Projecting {", ".join(map(str, sources))} into {area}')

        prev_inputs: List[float] = self._calc_prev_inputs(area, sources)
        input_sizes: Dict[BrainPart, int] = LazyConnectome._calculate_input_sizes(sources)
        total_k = sum(input_sizes.values())
        potential_winners = self._calc_potential_winners(area, total_k)
        first_winner_inputs, new_winners = self._calc_new_winners(area, prev_inputs, potential_winners)
        new_support_size = area.support_size + len(first_winner_inputs)
        return input_sizes, first_winner_inputs, new_winners, new_support_size

    # This class is not supposed to be used outside of this class, DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING!
    # =====================================================================================
    # =====================****====================================*****===================
    # ======================= *********** =============== ************ ====================
    # ========================== ************ ======== ************ =======================
    # ============================|   X    | =========== |   X   | ========================
    # ============================ --------- =========== -------- =========================
    # ===edoarad===========================================================================
    # ========================================= @@@@@@@@@ =================================
    # =====================================@@@@==========@@@@==============================
    # ====================================@@================@@=============================
    # ====================================@===================@============================
    # =====================================================================================
    # =======banana=================== | ==================================================
    # ================================ * ==================================================
    # =====================banana===== ** =================================================
    # ================================ *** =============banana=============================
    # ================================= **** ============================banana============
    # ================================== ***** ============================================
    # ===========banana================== ****** ==========================================
    # ===================================== ******* =======================================
    # ================banana================= ******* =================banana==============
    # =========================================== ***** ===================================
    # ================================================ ** =================================
    # =======================================================banana========================
    # ===============/----
    # ==============| ----
    class AreaInfo(NamedTuple):
        area: Area
        input_sizes: Dict[BrainPart, int]
        first_winner_inputs: List[float]
        new_winners: ndarray
        new_support_size: int

    def project(self, connections: Dict[BrainPart, List[Area]]):
        """ Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.
        :param connections A dictionary of connections to use in the projection, for example {area1
        """
        sources_mapping: defaultdict[Area, List[BrainPart]] = defaultdict(lambda: [])

        for part, areas in connections.items():
            for area in areas:
                sources_mapping[area] = sources_mapping[area] or []
                sources_mapping[area].append(part)
        # to_update is the set of all areas that receive input
        to_update = sources_mapping.keys()

        areas_info: List[LazyConnectome.AreaInfo] = \
            [self.AreaInfo(area, *self._project_into(area, sources_mapping[area])) for area in to_update]

        self.update_connectomes(areas_info)

        # once done everything, update areas winners and support size
        for area_info in areas_info:
            area_info.area.winners = area_info.new_winners
            area_info.area.support_size = area_info.new_support_size

    def _build_stim_to_area(self,
                            stim: Stimulus,
                            area: Area,
                            first_winner_to_inputs: Dict[int, Dict[BrainPart, int]]):
        # resize connections stim->area to the new support size
        self.connections[stim, area].synapses = np.resize(self.connections[stim, area].synapses,
                                                          (stim.n, area.support_size + len(first_winner_to_inputs)))
        # connections["first winner"] = how many fired from stim to this first winner
        for neuron in first_winner_to_inputs:
            self.connections[stim, area][area.support_size + neuron] = \
                first_winner_to_inputs[neuron][stim]

    def _build_fired_other_to_area(self,
                                   other: Area,
                                   area: Area,
                                   first_winner_to_inputs: Dict[int, Dict[BrainPart, int]]):
        # add num_first_winners columns to the connections
        self.connections[other, area].synapses = np.pad(self.connections[other, area].synapses,
                                                        ((0, 0), (0, len(first_winner_to_inputs))),
                                                        'constant', constant_values=0)
        for neuron in first_winner_to_inputs:
            # total_in - how many fired from from_area to this first winner (i)
            total_in = first_winner_to_inputs[neuron][other]
            # randomize which winners in from_area fired to i
            sample_indices = random.sample(other.winners, int(total_in))
            for other_neuron in range(other.support_size):
                # other_neuron that fired has connections with weight 1 (in prob 1)
                if other_neuron in sample_indices:
                    self.connections[other, area][other_neuron][area.support_size + neuron] = 1
                # other_neuron that is not winner has ??? weight 1 in prob p
                if other_neuron not in other.winners:
                    self.connections[other, area][other_neuron][area.support_size + neuron] = \
                        np.random.binomial(1, self.p)
                # other_neuron that is a winner and did not fire has connection 0 (since otherwise, it would fire)

    def _build_nonfired_other_to_area(self,
                                      other: Area,
                                      area: Area, new_support_size: int,
                                      first_winner_to_inputs: Dict[int, Dict[BrainPart, int]]):
        # add num_first_winners columns to self.connections[other_area, area]
        self.connections[other, area].synapses = np.pad(self.connections[other, area].synapses,
                                                        ((0, 0), (0, len(first_winner_to_inputs))), 'constant',
                                                        constant_values=0)

        for other_neuron in range(other.support_size):
            for neuron in range(area.support_size, new_support_size):
                # for all new neurons in support, add connections from other_area with weight 1 in prob p
                self.connections[other, area][other_neuron][neuron] = np.random.binomial(1, self.p)

    def _build_area_to_other(self,
                             other: Area,
                             area: Area,
                             new_support_size: int,
                             first_winner_to_inputs: Dict[int, Dict[BrainPart, int]]):
        self.connections[area, other].synapses = np.pad(self.connections[area, other].synapses,
                                                        ((0, len(first_winner_to_inputs)), (0, 0)), 'constant',
                                                        constant_values=0)
        columns = len(self.connections[area, other][0])
        for neuron in range(area.support_size, new_support_size):
            for other_neuron in range(columns):
                self.connections[area, other][neuron][other_neuron] = np.random.binomial(1, self.p)

    def _build_area_connections(self,
                                area_info: AreaInfo,
                                first_winner_to_inputs: Dict[int, Dict[BrainPart, int]]) -> None:
        if len(first_winner_to_inputs) == 0:
            return

        area, sources, new_winners, new_support_size = area_info.area, list(area_info.input_sizes.keys()), \
                                                       area_info.new_winners, area_info.new_support_size

        for source in sources:
            if isinstance(source, Stimulus):
                self._build_stim_to_area(source, area, first_winner_to_inputs)

            if isinstance(source, Area):
                self._build_fired_other_to_area(source, area, first_winner_to_inputs)

            beta = source.beta if isinstance(source, Area) else area.beta
            for i in new_winners:
                # update weight (*(1+beta)) for all neurons in stimulus / the winners in area
                source_neurons: Iterable[int] = range(source.n) if isinstance(source, Stimulus) else source.winners
                for j in source_neurons:
                    try:
                        self.connections[source, area][j][i] *= (1 + beta)
                    except BaseException as e:
                        print(f"Error is {e}")
                        print(f"connections are {self.connections[source, area][j]}")
                        raise e

        for other in self.areas:
            # expand the other_area->area connections for areas that did not fire
            if other not in sources:
                self._build_nonfired_other_to_area(other, area, new_support_size, first_winner_to_inputs)

            # expand the area->other_area connections for all areas
            # for all new neurons in support, add connections to other areas with weight 1 in prob p
            self._build_area_to_other(other, area, new_support_size, first_winner_to_inputs)

            logging.debug(f'Updated connection: '
                          f'{self.connections[area, other]}')

    @staticmethod
    def calculate_first_winner_to_inputs(first_winner_inputs: List[float], input_sizes: Dict[BrainPart, int]) -> \
            Dict[int, Dict[BrainPart, int]]:
        """
        Calculates first_winner_to_inputs
        first_winner_to_inputs := for each first winner i, first_winner_to_inputs[i] is a list of the number
        of inputs from each stimuli / area, randomly generated
        :param first_winner_inputs:
        :param input_sizes: a list containing all stimuli sizes, followed by all incoming areas winner counts
        :returns: first_winner_to_inputs
        """
        # for i in num_first_winners
        # generate where input came from
        # 	1) can sample input from array of size total_k, use ranges
        # 	2) can use stars/stripes method: if m total inputs, sample (m-1) out of total_k
        first_winner_to_inputs: Dict[int, Dict[BrainPart: int]] = {}
        for i in range(len(first_winner_inputs)):
            # first_winner_inputs[i] - how many fired into first winner # i
            # we randomize the indices that fired
            input_indices = random.sample(range(0, sum(input_sizes.values())), int(first_winner_inputs[i]))
            # inputs := a randomized array of the input size from each stimulus / area
            inputs: Dict[BrainPart: int] = {}  # np.zeros(len(input_sizes))
            total_so_far = 0
            for source in input_sizes:
                # divide the random indices to the different inputs, each input receives an amount of
                # input indices proportional to its size ("on average")
                inputs[source] = sum(
                    [(total_so_far <= w < (total_so_far + input_sizes[source])) for w in input_indices])
                total_so_far += input_sizes[source]
            first_winner_to_inputs[i] = inputs
            logging.debug(f'for first_winner #{i} with input {first_winner_inputs[i]} split as so: {inputs}')
        return first_winner_to_inputs

    def update_connectomes(self, area_infos: List['LazyConnectome.AreaInfo']) -> None:
        for area_info in area_infos:
            self._build_area_connections(area_info,
                                         LazyConnectome.calculate_first_winner_to_inputs(
                                             area_info.first_winner_inputs,
                                             area_info.input_sizes)
                                         )
