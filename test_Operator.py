from unittest import TestCase

from Operator import OperatorFunctions
from WeightMatrix import*
from Condition import*

class TestOperatorFunctions(TestCase):
    def test_choose_operator_case_condition_0(self):
        weights = WeightMatrix(2, 4)
        for i in range(30):
            operator = OperatorFunctions.choose_operator(weights=weights.weights, condition=Condition.C_0)
            self.assertTrue(operator.value >= 3)
            self.assertFalse(operator.value < 3)

    def test_apply_op(self):
        self.assertTrue(True) #TODO: Implement maybe

    def test_best_cost_route_crossover(self):
        self.fail()

    def test_find_best_task_position(self):
        self.fail()

    def test_intra_depot_removal(self):
        self.fail()

    def test_intra_depot_swapping(self):
        self.fail()

    def test_inter_depot_swapping(self):
        self.fail()

    def test_single_action_rerouting(self):
        self.fail()

    def test_one_move(self):
        self.fail()

    def test_two_swap(self):
        self.fail()
