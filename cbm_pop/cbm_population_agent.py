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

class CBMPopulationAgent(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_tasks, num_tsp_agents, num_iterations,
                 num_solution_attempts, agent_id, node_name: str):
        super().__init__(node_name)
        self.pop_size = pop_size  # Population size
        self.eta = eta  # Reinforcement learning factor
        self.rho = rho  # Mimetism rate
        # Number of cycles before changing exploration origin
        self.di_cycle_length = di_cycle_length
        self.num_iterations = num_iterations # Stopping criteria
        self.epsilon = epsilon  # Minimal solution improvement
        self.num_tasks = num_tasks  # Number of tasks
        self.num_tsp_agents = num_tsp_agents  # Number of agents
        self.agent_best_solution = None  # Best solution found by the agent
        self.coalition_best_solution = None  # Best found solution
        self.num_intensifiers = 2
        self.num_diversifiers = 4
        # Initialize population, weight matrix, and experience memory
        self.population = self.generate_population()
        self.cost_matrix = self.generate_problem()
        self.current_solution = self.select_solution()
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers)
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id
        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(String, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            String, 'weight_update', self.solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(String, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            String, 'weight_update', self.weight_update_callback, 10)


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

    def broadcast_solution(self, c_new):
        # Broadcast the solution (placeholder, no actual communication in this simple example)
        pass

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

    def receive_weight_matrix(self):
        # Placeholder for receiving a weight matrix from a neighboring agent, if available
        return None

    def weight_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        self.get_logger().info(f"Received weight update: {msg.data}")

    def solution_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        self.get_logger().info(f"Received weight update: {msg.data}")

    async def run(self):
        di_cycle_count = 0
        iteration_count = 0
        best_coalition_improved = False
        best_solution_value = Fitness.fitness_function(self.current_solution, self.cost_matrix)
        no_improvement_attempt_count = 0
        while not self.stopping_criterion(iteration_count):
            # Calculate the current state
            condition = ConditionFunctions.perceive_condition(self.previous_experience)

            # Check for minimal improvement in solution over n_cycles
            if no_improvement_attempt_count >= self.no_improvement_attempts:
                self.current_solution = self.select_random_solution()
                no_improvement_attempt_count = 0  # Reset cycle count

            # Choose and apply an operator
            operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition)
            c_new = OperatorFunctions.apply_op(
                operator,
                self.current_solution,
                self.population,
                self.cost_matrix)

            # Update experience history
            gain = Fitness.fitness_function(c_new, self.cost_matrix) - \
                   Fitness.fitness_function(self.current_solution, self.cost_matrix)
            self.update_experience(condition, operator, gain)

            print(f"Agent {self.agent_ID}: {Fitness.fitness_function(self.current_solution, self.cost_matrix)}")

            if self.coalition_best_solution is None or \
                    Fitness.fitness_function(c_new, self.cost_matrix) < Fitness.fitness_function(
                self.coalition_best_solution, self.cost_matrix):
                self.coalition_best_solution = deepcopy(c_new)
                best_coalition_improved = True

            # Update the best solution fitness after the individual learning
            if Fitness.fitness_function(c_new, self.cost_matrix) < best_solution_value:
                best_solution_value = Fitness.fitness_function(c_new, self.cost_matrix)
                no_improvement_attempt_count = 0
            else:
                no_improvement_attempt_count += 1

            self.current_solution = c_new

            di_cycle_count += 1  # Increment cycle count

            # Learning mechanisms at the end of a Diversification-Intensification (D-I) cycle
            if self.end_of_di_cycle(di_cycle_count):
                if best_coalition_improved:
                    self.weight_matrix.weights = self.individual_learning()
                    best_coalition_improved = False
                self.previous_experience = []
                di_cycle_count = 0


            iteration_count += 1

    def select_random_solution(self):
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

def main(args=None):
    rclpy.init(args=args)

    # Create and run the agent node
    node_name = "cbm_population_agent"
    agent = CBMPopulationAgent(
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_tasks=20, num_tsp_agents=5, num_iterations=1000,
        num_solution_attempts=20, agent_id=1, node_name=node_name
    )

    executor = MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
