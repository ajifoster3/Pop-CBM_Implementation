import random
from copy import deepcopy
from enum import Enum

import numpy as np


class Operator(Enum):
    TWO_SWAP = 1
    ONE_MOVE = 2
    BEST_COST_ROUTE_CROSSOVER = 3
    INTRA_DEPOT_REMOVAL = 4
    INTRA_DEPOT_SWAPPING = 5
    INTER_DEPOT_SWAPPING = 6
    SINGLE_ACTION_REROUTING = 7

class Condition(Enum):
    C_0 = 0
    C_1 = 1
    C_2 = 2
    C_3 = 3
    C_4 = 4

class CBM_PopulationAgent:
    def __init__(self,
                 pop_size,
                 eta,
                 rho,
                 n_cycles,
                 epsilon,
                 num_tasks,
                 num_agents,
                 num_operators,
                 num_states):
        self.pop_size = pop_size  # Population size
        self.eta = eta  # Reinforcement learning factor
        self.rho = rho  # Mimetism rate
        # Number of cycles before changing exploration origin
        self.n_cycles = n_cycles
        self.epsilon = epsilon  # Minimal solution improvement
        self.num_tasks = num_tasks  # Number of tasks
        self.num_agents = num_agents  # Number of agents
        self.agent_best_solution = None  # Best solution found by the agent
        self.coalition_best_solution = None  # Best found solution
        self.num_intensifiers = 2
        self.num_diversifiers = 5
        # Initialize population, weight matrix, and experience memory
        self.P = self.generate_population(self.pop_size,
                                          self.num_tasks,
                                          self.num_agents)
        self.cost_matrix = self.generate_problem(self.num_tasks)
        self.evaluate_population(self.P, self.cost_matrix)
        self.current_solution = self.select_solution(self.P, self.cost_matrix)
        self.W = self.init_weight_matrix()
        self.H = self.init_experience_memory()
        self.operator_function_map = {
            Operator.TWO_SWAP: self.two_swap,
            Operator.ONE_MOVE: self.one_move,
            Operator.BEST_COST_ROUTE_CROSSOVER: self.best_cost_route_crossover,
            Operator.INTRA_DEPOT_REMOVAL: self.intra_depot_removal,
            Operator.INTRA_DEPOT_SWAPPING: self.intra_depot_swapping,
            Operator.INTER_DEPOT_SWAPPING: self.inter_depot_swapping,
            Operator.SINGLE_ACTION_REROUTING: self.single_action_rerouting,
        }

    def generate_problem(self, number_tasks):
        """
        Randomly generates a problem of size `number_tasks`
        :param number_tasks: The number of tasks
        :return: Randomly generated symmetrical cost matrix representing the problem
        """
        # Set the random seed for reproducibility
        np.random.seed(0)

        # Generate a random symmetrical 20x20 cost matrix
        size = number_tasks
        cost_matrix = np.random.randint(1, 100, size=(size, size))

        # Make the matrix symmetrical
        cost_matrix = (cost_matrix + cost_matrix.T) // 2

        # Set the diagonal to zero (no cost for staying at the same location)
        np.fill_diagonal(cost_matrix, 0)
        return cost_matrix

    def generate_population(self, pop_size, number_tasks, num_agents):
        """
        Randomly generates a population of size `pop_size`
        :param pop_size: Size of the population
        :param number_tasks: number of tasks
        :param num_agents: number of agents
        :return: Population of solutions of size `pop_size`
        """
        # Generate initial population where each solution is a list of task allocations to agents
        population = []

        for _ in range(pop_size):
            # Create a list of task indexes and shuffle it for a random allocation
            allocation = list(range(number_tasks))
            random.shuffle(allocation)

            # Generate non-zero task counts for each agent that sum to number_tasks
            # Start with each agent assigned at least 1 task
            counts = [1] * num_agents
            for _ in range(number_tasks - num_agents):
                counts[random.randint(0, num_agents - 1)] += 1

            # Add both allocation and counts to the population
            population.append((allocation, counts))

        return population

    #TODO: Maybe remove this?
    def evaluate_population(self, population, cost_matrix):
        """
        Calculates the fitness of all solutions, Not sure necessary
        :param population:
        :param cost_matrix:
        :return:
        """
        # Evaluate each solution based on workload balance or other criteria
        for solution in population:
            # Calls the fitness function to evaluate each solution
            self.fitness_function(solution, cost_matrix)

    def select_solution(self, population, cost_matrix):
        """
        Finds and returns the fittest solution
        :param population: Population of solutions
        :param cost_matrix: Cost matrix
        :return: Fittest solution
        """
        # Select the best solution from the population based on fitness score
        best_solution = min(population, key=lambda sol: self.fitness_function(
            sol, cost_matrix))  # Assuming lower score is better
        return best_solution

    def init_weight_matrix(self):
        """
        Generates a weight matrix mapping conditions onto operations.
        0 0 1 1 1 1 1 <- Initial diversification operator
        1 1 0 0 0 0 0 <- Subsequent intensification operator
        0 1 0 0 0 0 0 <- Which operator after intensification operator 1
        1 0 0 0 0 0 0 <- Which operator after intensification operator 2 (Move rows if there's more intensifiers)
        0 0 1 1 1 1 1 <- If all intensification operators haven't improved the solution, which diversificator to use
        :return: A weight matrix mapping conditions to operations
        """
        # Initialize and return a weight matrix (for operator selection, if needed)
        weight_matrix = []
        initial_diversifier_condition_row = [0] * self.num_intensifiers + [1] * self.num_diversifiers
        weight_matrix.append(initial_diversifier_condition_row)
        initial_intensifier_condition_row = [1] * self.num_intensifiers + [0] * self.num_diversifiers
        weight_matrix.append(initial_intensifier_condition_row)
        for i in range(self.num_intensifiers):
            intensifier_condition_row = [1] * self.num_intensifiers + [0] * self.num_diversifiers
            intensifier_condition_row[i] = 0
            weight_matrix.append(intensifier_condition_row)
        final_diversifier_condition_row = [0] * self.num_intensifiers + [1] * self.num_diversifiers
        weight_matrix.append(final_diversifier_condition_row)
        return weight_matrix

    #TODO: Maybe remove this?
    def init_experience_memory(self):
        """
        Initialise the experience memory. Maybe unnecessary?
        :return: Empty experience memory
        """
        # Initialize and return experience memory
        return []

    def perceive_condition(self, H):
        """
        Calculates the condition based on the experience memory of which operators where used previously.
        :param H: Experience memory
        :return: The current condition
        """
        if not H:
            return Condition.C_0
        if H[-1][1] in {Operator.BEST_COST_ROUTE_CROSSOVER,
                        Operator.INTRA_DEPOT_REMOVAL,
                        Operator.INTRA_DEPOT_SWAPPING,
                        Operator.INTER_DEPOT_SWAPPING,
                        Operator.SINGLE_ACTION_REROUTING}:
            return Condition.C_1
        if H[-1][1] == Operator.TWO_SWAP:
            return Condition.C_2
        if H[-1][1] == Operator.ONE_MOVE:
            return Condition.C_3
        return Condition.C_4 # TODO: This needs changing to see if cost has decreased since applying intensifiers.

    def choose_operator(self, W, condition):
        """
        Stochastically choose an operator for a condition using the weights.
        :param W: Weights
        :param condition: Current condition
        :return: The chosen operator
        """
        # Choose an operator (e.g., mutation, crossover) based on weight matrix W and current state
        # For simplicity, we only apply a mutation operator in this example
        row = W[condition.value]
        operators = list(Operator)

        # Randomly select an operator based on weights in `row`
        chosen_operator = random.choices(operators, weights=row, k=1)[0]
        return chosen_operator

    def apply_op(self, operator, current_solution, population, coalition_best_solution=None):
        """
        Apply the operator to the current solution and return the newly generated child solution.
        :param operator: The operator to be applied
        :param current_solution: The current solution
        :param population: The population
        :param coalition_best_solution: The best solution among the coalition
        :return: A child solution
        """
        # Get the function based on the operator
        if operator in self.operator_function_map:
            # Call the function and pass arguments as needed
            if operator == Operator.BEST_COST_ROUTE_CROSSOVER:
                return self.operator_function_map[operator](current_solution, population)
            else:
                return self.operator_function_map[operator](current_solution)

        # Raise an exception if the operator is not recognized
        raise Exception("Something went wrong! The selected operation doesn't exist.")

    def update_experience(self, condition, operator, gain):
        """
        Adds details of the current iteration to the experience memory.
        :param condition: The previous condition
        :param operator: The operator applied
        :param gain: The resulting change in the current solution's fitness
        :return: None
        """
        self.H.append([condition, operator, gain])
        pass

    def fitness_function(self, solution, cost_matrix):
        """
        Returns the fitness of the solution using the cost matrix.
        :param solution: Solution to be calculated
        :param cost_matrix: Cost matrix to calculate with
        :return: The fitness of the solution
        """
        # Extract the task order from the solution
        task_order, agent_task_counts = solution

        # Calculate the total cost based on the cost matrix
        total_cost = 0

        # Sum up the costs for the assigned tasks in the task order
        counter = 0
        for j in agent_task_counts:

            for i in range(counter, counter + j - 1):
                task_i = task_order[i]
                task_j = task_order[i + 1]
                total_cost += cost_matrix[task_i][task_j]
            counter += j

        return total_cost


    # TODO: Implement Learning!!!

    def broadcast_solution(self, C_new):
        # Broadcast the solution (placeholder, no actual communication in this simple example)
        pass

    def individual_learning(self, W, H, eta):
        # Update weight matrix (if needed) based on learning (not fully implemented in this example)
        return W

    def broadcast_weight_matrix(self, W):
        # Broadcast the weight matrix (placeholder, no actual communication in this example)
        pass

    def mimetism_learning(self, W, W_received, rho):
        # Mimetism learning: Combine W with W_received based on rho (placeholder in this example)
        return W

    # TODO: Implement termination criteria
    # TODO: Some of these methods maybe unnecessary

    def stopping_criterion(self):
        # Define a stopping criterion (e.g., a fixed number of iterations)
        return False  # Placeholder; replace with actual condition


    def no_improvement_in_best_solution(self):
        # Check if there has been no change in the best solution greater than epsilon
        return False  # Placeholder; replace with actual condition

    def end_of_DI_cycle(self, cycle_count, n_cycles):
        if cycle_count >= n_cycles:
            return True
        return False  # Placeholder; replace with actual condition

    def receive_weight_matrix(self):
        # Placeholder for receiving a weight matrix from a neighboring agent, if available
        return None

    # Diversifiers

    def best_cost_route_crossover(self, current_solution, P):
        """
        "For two parent chromosomes, select a route to be removed
        from each. The removed nodes are inserted into the
        other parent solution at the best insertion cost"
        :param P: Population
        :param current_solution: The current solution as a parent
        :return: A child solution
        """
        return None

    def intra_depot_removal(self, current_solution):
        """
        "Two cutpoints in the chromosome  associated with the
        robot initial position are selected and the genetic
        material between these two cutpoints is reversed."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """
        return None

    def intra_depot_swapping(self, current_solution):
        """
        "This simple mutation operator selects two random routes from
        the same initial position and exchanges a randomly selected
        action from one route to another."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """
        return None

    def inter_depot_swapping(self, current_solution):
        """
        "Mutation of swapping nodes in the routes of different initial
        positions. Candidates for this mutation are nodes that are in
        similar proximity to more than one initial position."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """
        return None

    def single_action_rerouting(self, current_solution):
        """
        "Re-routing involves randomly selecting one action and removing
        it from the existing route. The action is then inserted at the
        best feasible insertion point within the entire chromosome."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """
        return None

    # Intensifiers

    def two_swap(self, current_solution):
        """
        "Swapping of borderline actions from two initial positions to
        improve solution fitness"
        :param current_solution: The current solution to be optimised
        :return: A child solution
        """
        return None

    def one_move(self, current_solution):
        """
        "Removal of a node from the solution and insertion at the point
        that maximizes solution fitness"
        :param current_solution: The current solution to be optimised
        :return: A child solution
        """

    def run(self):
        cycle_count = 0
        previous_state = None;
        best_coalition_improved = False
        while not self.stopping_criterion():
            # Calculate the current state
            condition = self.perceive_condition(self.H)

            # Check for minimal improvement in solution over n_cycles
            if cycle_count >= self.n_cycles and self.no_improvement_in_best_solution():
                self.evaluate_population(self.P, self.cost_matrix)
                self.current_solution = self.select_solution(self.P)
                cycle_count = 0  # Reset cycle count

            # Choose and apply an operator
            operator = self.choose_operator(self.W, condition)
            C_new = self.apply_op(operator, self.current_solution,
                                  self.P, self.coalition_best_solution)

            # Update experience history
            gain = self.fitness_function(self.current_solution, self.cost_matrix) - \
                   self.fitness_function(C_new, self.cost_matrix)
            self.update_experience(self.H, condition, operator, gain)

            # Update solutions if there is an improvement in coallition_best_solution
            if self.coalition_best_solution is None or self.fitness_function(C_new, self.cost_matrix) < self.fitness_function(self.coalition_best_solution,
                                                                                                                              self.cost_matrix):
                self.coalition_best_solution = deepcopy(C_new)
                best_coalition_improved = True

            # Learning mechanisms at the end of a Diversification-Intensification (D-I) cycle
            if self.end_of_DI_cycle(cycle_count, self.n_cycles):
                if best_coalition_improved:
                    self.W = self.individual_learning(self.W, self.H, self.eta)

                # Mimetism learning if weight matrix is received from a neighbor
                #W_received = self.receive_weight_matrix()
                #if W_received:
                #    self.W = self.mimetism_learning(self.W, W_received, self.rho)

                cycle_count = 0;
                previous_state = self.H[-1][1]
                self.H = []

            cycle_count += 1  # Increment cycle count


if __name__ == '__main__':
    cbm = CBM_PopulationAgent(20, 0.5, 1, 5, 0.5, 10, 2, 5, 5)
    cbm.run()
