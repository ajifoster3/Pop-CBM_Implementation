import random
import numpy as np
from copy import deepcopy
from random import sample
from cbm_pop.Condition import ConditionFunctions
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
from cbm_pop_interfaces.msg import Solution, Weights

class CBMPopulationAgent(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_tasks, num_tsp_agents, num_iterations,
                 num_solution_attempts, agent_id, node_name: str):
        super().__init__(node_name)
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.num_tasks = num_tasks
        self.num_tsp_agents = num_tsp_agents
        self.agent_best_solution = None
        self.coalition_best_solution = None
        self.local_best_solution = None
        self.coalition_best_agent = None
        self.num_intensifiers = 2
        self.num_diversifiers = 4
        self.population = self.generate_population()
        self.cost_matrix = self.generate_problem()
        self.current_solution = self.select_solution()
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers)
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.best_solution_value = Fitness.fitness_function(self.current_solution, self.cost_matrix)
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.weight_update_callback, 10)

        # Timer for periodic execution of the run loop
        self.run_timer = self.create_timer(0.1, self.run_step)


    def generate_problem(self):
        """
        Randomly generates a problem of size `number_tasks`
        :return: Randomly generated symmetrical cost matrix representing the problem
        """
        # Set the random seed for reproducibility
        np.random.seed(0)

        # Generate a random symmetrical 20x20 cost matrix
        size = self.num_tasks
        cost_matrix = np.random.randint(1, 100, size=(size, size))

        # Make the matrix symmetrical
        cost_matrix = (cost_matrix + cost_matrix.T) // 2

        # Set the diagonal to zero (no cost for staying at the same location)
        np.fill_diagonal(cost_matrix, 0)
        return cost_matrix

    def generate_population(self):
        """
        Randomly generates a population of size `pop_size`
        :return: Population of solutions of size `pop_size`
        """
        # Generate initial population where each solution is a list of task allocations to agents
        population = []

        for _ in range(self.pop_size):
            # Create a list of task indexes and shuffle it for a random allocation
            allocation = list(range(self.num_tasks))
            random.shuffle(allocation)

            # Generate non-zero task counts for each agent that sum to number_tasks
            # Start with each agent assigned at least 1 task
            counts = [1] * self.num_tsp_agents
            for _ in range(self.num_tasks - self.num_tsp_agents):
                counts[random.randint(0, self.num_tsp_agents - 1)] += 1

            # Add both allocation and counts to the population
            population.append((allocation, counts))

        return population

    def select_solution(self):
        """
        Finds and returns the fittest solution
        :return: The fittest solution
        """
        # Select the best solution from the population based on fitness score
        best_solution = min(self.population, key=lambda sol: Fitness.fitness_function(
            sol, self.cost_matrix))  # Assuming lower score is better
        return best_solution

    def update_experience(self, condition, operator, gain):
        """
        Adds details of the current iteration to the experience memory.
        :param condition: The previous condition
        :param operator: The operator applied
        :param gain: The resulting change in the current solution's fitness
        :return: None
        """
        self.previous_experience.append([condition, operator, gain])
        pass

    # TODO: Implement Learning!!!

    def individual_learning(self):
        # Update weight matrix (if needed) based on learning (not fully implemented in this example)
        abs_gain = 0
        index_best_fitness = -1
        for i in range(len(self.previous_experience)):
            current_gain = abs_gain + self.previous_experience[i][2]
            if current_gain < abs_gain:
                index_best_fitness = i
            abs_gain += current_gain

        # Get elements before index_best_fitness
        elements_before_best = self.previous_experience[:index_best_fitness+1] if index_best_fitness != -1 else []
        condition_operator_pairs = [(item[0], item[1]) for item in elements_before_best]
        condition_operator_pairs = list(set(condition_operator_pairs))
        for pair in condition_operator_pairs:
            self.weight_matrix.weights[pair[0].value][pair[1].value-1] += self.eta # TODO: Eta2 for beating coalition fitness
        return self.weight_matrix.weights

    def broadcast_weight_matrix(self, weights):
        # Broadcast the weight matrix (placeholder, no actual communication in this example)
        pass

    def mimetism_learning(self, received_weights, rho):
        # Mimetism learning: Combine W with W_received based on rho (placeholder in this example)

        return self.weight_matrix.weights

    def stopping_criterion(self, iteration_count):
        # Define a stopping criterion (e.g., a fixed number of iterations)
        return iteration_count > self.num_iterations

    def end_of_di_cycle(self, cycle_count):
        if cycle_count >= self.di_cycle_length:
            return True
        return False  # Placeholder; replace with actual condition

    def weight_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        self.get_logger().info(f"Received weight update: ")

    def solution_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        solution = (msg.order, msg.allocations)
        if Fitness.fitness_function(solution, self.cost_matrix) > self.best_solution_value:
            self.best_solution_value = Fitness.fitness_function(solution, self.cost_matrix)
            self.coalition_best_solution = solution
            self.coalition_best_agent = msg.id

    def select_random_solution(self):
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """
        if self.stopping_criterion(self.iteration_count):
            self.get_logger().info("Stopping criterion met. Shutting down.")
            self.run_timer.cancel()
            return

        condition = ConditionFunctions.perceive_condition(self.previous_experience)

        if self.no_improvement_attempt_count >= self.no_improvement_attempts:
            self.current_solution = self.select_random_solution()
            self.no_improvement_attempt_count = 0

        operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition)
        c_new = OperatorFunctions.apply_op(
            operator,
            self.current_solution,
            self.population,
            self.cost_matrix
        )

        gain = Fitness.fitness_function(c_new, self.cost_matrix) - \
               Fitness.fitness_function(self.current_solution, self.cost_matrix)
        self.update_experience(condition, operator, gain)

        if self.local_best_solution is None or \
                Fitness.fitness_function(c_new, self.cost_matrix) < Fitness.fitness_function(
                    self.local_best_solution, self.cost_matrix):
            self.local_best_solution = deepcopy(c_new)
            self.best_local_improved = True

        if self.coalition_best_solution is None or \
                Fitness.fitness_function(c_new, self.cost_matrix) < Fitness.fitness_function(
                    self.coalition_best_solution, self.cost_matrix):
            self.coalition_best_solution = deepcopy(c_new)
            self.coalition_best_agent = self.agent_ID
            self.best_coalition_improved = True

            solution = Solution()
            solution.id = self.agent_ID
            solution.order = self.coalition_best_solution[0]
            solution.allocations = self.coalition_best_solution[1]
            self.solution_publisher.publish(solution)

        if Fitness.fitness_function(c_new, self.cost_matrix) < self.best_solution_value:
            self.best_solution_value = Fitness.fitness_function(c_new, self.cost_matrix)
            self.no_improvement_attempt_count = 0
        else:
            self.no_improvement_attempt_count += 1

        self.current_solution = c_new
        self.di_cycle_count += 1

        if self.end_of_di_cycle(self.di_cycle_count):
            if self.best_local_improved:
                self.weight_matrix.weights = self.individual_learning()
                self.best_local_improved = False

            if self.best_coalition_improved:
                self.weight_matrix.weights = self.individual_learning()
                self.best_coalition_improved = False
                # TODO: Broadcast weight matrix

            self.previous_experience = []
            self.di_cycle_count = 0

        self.iteration_count += 1
        self.get_logger().info(f"Iteration {self.iteration_count}: Current best solution fitness = {self.best_solution_value}")


def main(args=None):
    rclpy.init(args=args)

    # Create and run the agent node
    node_name = "cbm_population_agent"
    agent = CBMPopulationAgent(
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_tasks=200, num_tsp_agents=5, num_iterations=1000,
        num_solution_attempts=20, agent_id=1, node_name=node_name
    )

    try:
        rclpy.spin(agent)  # Run the ROS2 executor
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()