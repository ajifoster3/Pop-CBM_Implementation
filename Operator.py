import random
from copy import deepcopy
from enum import Enum

from Fitness import Fitness


class Operator(Enum):
    TWO_SWAP = 1
    ONE_MOVE = 2
    BEST_COST_ROUTE_CROSSOVER = 3
    INTRA_DEPOT_REMOVAL = 4
    INTRA_DEPOT_SWAPPING = 5
    #INTER_DEPOT_SWAPPING = 6 Not applicable due to lack of depots
    SINGLE_ACTION_REROUTING = 6

class OperatorFunctions:
    # Define operator function map with static method references
    operator_function_map = {
        Operator.TWO_SWAP: lambda current_solution, cost_matrix: OperatorFunctions.two_swap(current_solution, cost_matrix),
        Operator.ONE_MOVE: lambda current_solution, cost_matrix: OperatorFunctions.one_move(current_solution, cost_matrix),
        Operator.BEST_COST_ROUTE_CROSSOVER: lambda current_solution,
                                                   population,
                                                   cost_matrix: OperatorFunctions.best_cost_route_crossover(
            current_solution, population, cost_matrix),
        Operator.INTRA_DEPOT_REMOVAL: lambda current_solution: OperatorFunctions.intra_depot_removal(current_solution),
        Operator.INTRA_DEPOT_SWAPPING: lambda current_solution: OperatorFunctions.intra_depot_swapping(
            current_solution),
        #Operator.INTER_DEPOT_SWAPPING: lambda current_solution: OperatorFunctions.inter_depot_swapping(
        #    current_solution),
        Operator.SINGLE_ACTION_REROUTING: lambda current_solution, cost_matrix: OperatorFunctions.single_action_rerouting(current_solution, cost_matrix)
    }


    def choose_operator(W, condition):
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

    def apply_op(operator, current_solution, population, coalition_best_solution=None, cost_matrix=None):
        """
        Apply the operator to the current solution and return the newly generated child solution.
        :param cost_matrix: Cost matrix to calculate the cost
        :param operator: The operator to be applied
        :param current_solution: The current solution
        :param population: The population
        :param coalition_best_solution: The best solution among the coalition
        :return: A child solution
        """
        # Get the function based on the operator
        if operator in OperatorFunctions.operator_function_map:
            # Call the function and pass arguments as needed
            if operator == Operator.BEST_COST_ROUTE_CROSSOVER:
                return OperatorFunctions.operator_function_map[operator](current_solution, population, cost_matrix)
            elif (operator == Operator.SINGLE_ACTION_REROUTING
                  or operator == Operator.TWO_SWAP
                  or operator == Operator.ONE_MOVE):
                    return OperatorFunctions.operator_function_map[operator](current_solution, cost_matrix)
            else:
                return OperatorFunctions.operator_function_map[operator](current_solution)

        # Raise an exception if the operator is not recognized
        raise Exception("Something went wrong! The selected operation doesn't exist.")

    # Diversifiers
    @staticmethod
    def best_cost_route_crossover(current_solution, P, costMatrix):
        """
        "For two parent chromosomes, select a route to be removed
        from each. The removed nodes are inserted into the
        other parent solution at the best insertion cost."
        At the moment it selects the best solution in P
        other than the current solution.
        :param P: Population
        :param current_solution: The current solution as a parent
        :return: A child solution
        """
        # Find the fittest solution in P that is not current_solution
        fittest_non_current_solution = min(
            (sol for sol in P if sol != current_solution),
            key=lambda sol: Fitness.fitness_function(sol, costMatrix)
        )

        # Randomly select a path (route) from fittest_non_current_solution
        task_order, agent_task_counts = fittest_non_current_solution
        selected_agent = random.randint(0, len(agent_task_counts) - 1)

        # Identify the start and end indices for the selected agent's path
        start_index = sum(agent_task_counts[:selected_agent])
        end_index = start_index + agent_task_counts[selected_agent]

        # Extract the path for the selected agent
        selected_path = task_order[start_index:end_index]

        # Create a copy of current_solution to modify
        new_solution_task_order, new_solution_task_counts = deepcopy(current_solution)

        temp_count_counter = 0
        temp_task_counter = 0
        # For each agent
        for i in new_solution_task_counts:
            # For each task for that agent
            for j in range(i):
                if new_solution_task_order[temp_task_counter] in selected_path:
                    new_solution_task_order.remove(new_solution_task_order[temp_task_counter])
                    new_solution_task_counts[temp_count_counter] -= 1
                    temp_task_counter -= 1
                temp_task_counter += 1
            temp_count_counter += 1

        for task in selected_path[:]:  # Use a copy of selected_path to iterate safely
            best_fitness = float('inf')
            best_position = 0
            best_agent = 0

            # Try inserting the task at each position in the task order
            for agent_index, count in enumerate(new_solution_task_counts):
                # Calculate the insertion range for this agent
                agent_start_index = sum(new_solution_task_counts[:agent_index])
                agent_end_index = agent_start_index + count

                # Try inserting within this agent's range
                for pos in range(agent_start_index, agent_end_index + 1):
                    temp_order = new_solution_task_order[:]
                    temp_order.insert(pos, task)

                    # Update task counts temporarily for fitness calculation
                    temp_counts = new_solution_task_counts[:]
                    temp_counts[agent_index] += 1

                    # Calculate fitness with this temporary insertion
                    temp_fitness = Fitness.fitness_function((temp_order, temp_counts), costMatrix)

                    # If the new fitness is better, update best fitness, position, and agent
                    if temp_fitness < best_fitness:
                        best_fitness = temp_fitness
                        best_position = pos
                        best_agent = agent_index

            # Insert the task at the best position found and update the task count for that agent
            new_solution_task_order.insert(best_position, task)
            new_solution_task_counts[best_agent] += 1

        # Return the modified solution as the child solution
        return new_solution_task_order, new_solution_task_counts # TODO: new_solution_task_counts came out as [6,6] for a 10 task problem

    def intra_depot_removal(current_solution):
        """
        "Two cutpoints in the chromosome associated with the
        robot initial position are selected and the genetic
        material between these two cutpoints is reversed."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """
        # Extract task order and agent task counts from current solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Randomly select an agent
        selected_agent = random.randint(0, len(agent_task_counts) - 1)

        # Calculate the start and end index for this agent's route
        start_index = sum(agent_task_counts[:selected_agent])
        end_index = start_index + agent_task_counts[selected_agent]

        # Perform the reversal mutation if the agent has enough tasks
        if end_index - start_index > 1:  # Ensure there are enough tasks to reverse a section
            # Randomly choose two cut points within this range
            cut1, cut2 = sorted(random.sample(range(start_index, end_index), 2))

            # Reverse the section between the two cut points
            task_order[cut1:cut2 + 1] = reversed(task_order[cut1:cut2 + 1])
        return task_order, agent_task_counts

    def intra_depot_swapping(current_solution):
        """
        Perform an intra-depot mutation by selecting two random routes
        from the solution and moving a randomly selected task from
        one route to another.
        :param current_solution: The current solution to be mutated
        :return: A mutated child solution
        """
        # Extract task order and agent task counts from the current solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Randomly select two distinct agents (routes) to swap between
        agent1, agent2 = random.sample(range(len(agent_task_counts)), 2)

        # Determine the task range for each agent
        start_index1 = sum(agent_task_counts[:agent1]) + 1
        end_index1 = start_index1 + agent_task_counts[agent1] - 1

        start_index2 = sum(agent_task_counts[:agent2])
        end_index2 = start_index2 + agent_task_counts[agent2]

        # Ensure the selected agent has tasks to swap
        if end_index1 > start_index1:
            # Randomly select a task from agent1's route
            task_index = random.randint(start_index1, end_index1 - 1)
            task = task_order.pop(task_index)
            agent_task_counts[agent1] -= 1

            # Insert the task into a random position in agent2's route
            if end_index2 > start_index2:
                insert_position = random.randint(start_index2, end_index2)
            else:
                insert_position = start_index2

            task_order.insert(insert_position, task)
            agent_task_counts[agent2] += 1

        # Return the modified solution
        return task_order, agent_task_counts

    def inter_depot_swapping(current_solution):
        """
        "Mutation of swapping nodes in the routes of different initial
        positions. Candidates for this mutation are nodes that are in
        similar proximity to more than one initial position."

        Only applicable with Depots.

        :param current_solution: The current solution to be mutated
        :return: A child solution
        """

        return current_solution

    def single_action_rerouting(current_solution, cost_matrix):
        """
        "Re-routing involves randomly selecting one action and removing
        it from the existing route. The action is then inserted at the
        best feasible insertion point within the entire chromosome."

        Confusing!

        :param current_solution: The current solution to be mutated
        :return: A modified solution with improved fitness
        """
        return current_solution
    # Intensifiers

    def two_swap(current_solution, cost_matrix):
        """
        "Swapping of borderline actions from two initial positions to
        improve solution fitness"
        :param current_solution: The current solution to be optimised
        :return: A child solution
        """
        return current_solution


    def one_move(current_solution, cost_matrix):
        """
        "Removal of a node from the solution and insertion at the point
        that maximizes solution fitness"
        :param current_solution: The current solution to be optimised
        :return: A child solution
        """

        # Deep copy to avoid modifying the original solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Select a random action (task) from the entire task order
        if not task_order:
            return current_solution  # Return unchanged if task_order is empty

        # Randomly select an action to remove
        task_index = random.randint(0, len(task_order) - 1)
        task = task_order.pop(task_index)

        # Adjust task count for the agent that lost the task
        agent_index = next(i for i, count in enumerate(agent_task_counts) if
                           sum(agent_task_counts[:i]) <= task_index < sum(agent_task_counts[:i + 1]))
        agent_task_counts[agent_index] -= 1

        # Initialize variables to track best insertion
        best_fitness = float('inf')
        best_position = 0
        best_agent = 0

        # Iterate over possible insertion positions
        for i, count in enumerate(agent_task_counts):
            start_index = sum(agent_task_counts[:i])
            end_index = start_index + count

            # Try inserting task at every possible position for the current agent
            for pos in range(start_index, end_index + 1):
                temp_order = task_order[:]
                temp_order.insert(pos, task)

                # Temporary counts for fitness calculation
                temp_counts = agent_task_counts[:]
                temp_counts[i] += 1

                # Calculate fitness
                temp_fitness = Fitness.fitness_function((temp_order, temp_counts), cost_matrix)

                # Check if this position yields better fitness
                if temp_fitness < best_fitness:
                    best_fitness = temp_fitness
                    best_position = pos
                    best_agent = i

        # Insert the task at the best position and update task counts
        task_order.insert(best_position, task)
        agent_task_counts[best_agent] += 1

        return task_order, agent_task_counts