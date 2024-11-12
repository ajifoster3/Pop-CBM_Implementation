import random
from copy import deepcopy
import Operator
import enum as Enum
import numpy as np
from Operator import OperatorFunctions
from Fitness import Fitness
from Condition import ConditionFunctions
from WeightMatrix import WeightMatrix

class CBMPopulationAgent:
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
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers)
        self.H = self.init_experience_memory()

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

    # TODO: Maybe remove this?
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
            Fitness.fitness_function(solution, cost_matrix)

    def select_solution(self, population, cost_matrix):
        """
        Finds and returns the fittest solution
        :param population: Population of solutions
        :param cost_matrix: Cost matrix
        :return: Fittest solution
        """
        # Select the best solution from the population based on fitness score
        best_solution = min(population, key=lambda sol: Fitness.fitness_function(
            sol, cost_matrix))  # Assuming lower score is better
        return best_solution

    # TODO: Maybe remove this?
    def init_experience_memory(self):
        """
        Initialise the experience memory. Maybe unnecessary?
        :return: Empty experience memory
        """
        # Initialize and return experience memory
        return []

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

    def run(self):
        cycle_count = 0
        previous_state = None;
        best_coalition_improved = False
        while not self.stopping_criterion():
            # Calculate the current state
            condition = ConditionFunctions.perceive_condition(self.H)

            # Check for minimal improvement in solution over n_cycles
            if cycle_count >= self.n_cycles and self.no_improvement_in_best_solution():
                self.evaluate_population(self.P, self.cost_matrix)
                self.current_solution = self.select_solution(self.P)
                cycle_count = 0  # Reset cycle count

            # Choose and apply an operator
            operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition)
            C_new = OperatorFunctions.apply_op(
                operator,
                self.current_solution,
                self.P,
                self.coalition_best_solution,
                self.cost_matrix)

            # Update experience history
            gain = Fitness.fitness_function(self.current_solution, self.cost_matrix) - \
                   Fitness.fitness_function(C_new, self.cost_matrix)
            self.update_experience(condition, operator, gain)

            # Update solutions if there is an improvement in coallition_best_solution
            if self.coalition_best_solution is None or Fitness.fitness_function(C_new,
                                                                             self.cost_matrix) < Fitness.fitness_function(
                self.coalition_best_solution,
                self.cost_matrix):
                self.coalition_best_solution = deepcopy(C_new)
                best_coalition_improved = True

            # Learning mechanisms at the end of a Diversification-Intensification (D-I) cycle
            if self.end_of_DI_cycle(cycle_count, self.n_cycles):
                if best_coalition_improved:
                    self.weight_matrix.weights = self.individual_learning(self.weight_matrix.weights, self.H, self.eta)

                # Mimetism learning if weight matrix is received from a neighbor
                # W_received = self.receive_weight_matrix()
                # if W_received:
                #    self.W = self.mimetism_learning(self.W, W_received, self.rho)

                cycle_count = 0;
                previous_state = self.H[-1][1]
                self.H = []

            cycle_count += 1  # Increment cycle count


if __name__ == '__main__':
    cbm = CBMPopulationAgent(20, 0.5, 1, 5, 0.5, 10, 2, 5, 5)
    cbm.run()
