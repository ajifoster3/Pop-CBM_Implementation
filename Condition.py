from Operator import Operator
from enum import Enum

import Operator

class Condition(Enum):
    C_0 = 0 # Starting a DI-cycle
    C_1 = 1 # A diversification operator was previous applied
    C_2 = 2 # The first intensification operator was applied
    C_3 = 3 # The second intensification operator was applied
    C_4 = 4 # All intensification operators have been applied this DI-cycle and fitness hasn't improved


class ConditionFunctions:
    @staticmethod
    def perceive_condition(H):
        """
        Calculates the condition based on the experience memory of which operators where used previously.
        :param H: Experience memory
        :return: The current condition
        """
        if not H:
            return Condition.C_0
        if H[-1][1] in {Operator.Operator.BEST_COST_ROUTE_CROSSOVER,
                        Operator.Operator.INTRA_DEPOT_REMOVAL,
                        Operator.Operator.INTRA_DEPOT_SWAPPING,
                        Operator.Operator.INTER_DEPOT_SWAPPING,
                        Operator.Operator.SINGLE_ACTION_REROUTING}:
            return Condition.C_1
        if H[-1][1] == Operator.Operator.TWO_SWAP:
            return Condition.C_2
        if H[-1][1] == Operator.Operator.ONE_MOVE:
            return Condition.C_3
        return Condition.C_4 # TODO: This needs changing to see if cost has decreased since applying intensifiers.
