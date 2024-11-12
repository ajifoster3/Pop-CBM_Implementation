from enum import Enum

class Operator(Enum):
    TWO_SWAP = 1
    ONE_MOVE = 2
    BEST_COST_ROUTE_CROSSOVER = 3
    INTRA_DEPOT_REMOVAL = 4
    INTRA_DEPOT_SWAPPING = 5
    INTER_DEPOT_SWAPPING = 6
    SINGLE_ACTION_REROUTING = 7

class OperatorFunctions:
    def __int__(self):
        self.operator_function_map = {
            Operator.TWO_SWAP: self.two_swap,
            Operator.ONE_MOVE: self.one_move,
            Operator.BEST_COST_ROUTE_CROSSOVER: self.best_cost_route_crossover,
            Operator.INTRA_DEPOT_REMOVAL: self.intra_depot_removal,
            Operator.INTRA_DEPOT_SWAPPING: self.intra_depot_swapping,
            Operator.INTER_DEPOT_SWAPPING: self.inter_depot_swapping,
            Operator.SINGLE_ACTION_REROUTING: self.single_action_rerouting,
        }
    # Diversifiers

    def best_cost_route_crossover(self, current_solution, P):
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
            key=lambda sol: self.fitness_function(sol, self.cost_matrix)
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
                    temp_fitness = self.fitness_function((temp_order, temp_counts), self.cost_matrix)

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

    def intra_depot_removal(self, current_solution):
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

    def intra_depot_swapping(self, current_solution):
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

    def inter_depot_swapping(self, current_solution):
        """
        "Mutation of swapping nodes in the routes of different initial
        positions. Candidates for this mutation are nodes that are in
        similar proximity to more than one initial position."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """

        return current_solution

    def single_action_rerouting(self, current_solution):
        """
        "Re-routing involves randomly selecting one action and removing
        it from the existing route. The action is then inserted at the
        best feasible insertion point within the entire chromosome."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """
        return current_solution

    # Intensifiers

    def two_swap(self, current_solution):
        """
        "Swapping of borderline actions from two initial positions to
        improve solution fitness"
        :param current_solution: The current solution to be optimised
        :return: A child solution
        """
        return current_solution

    def one_move(self, current_solution):
        """
        "Removal of a node from the solution and insertion at the point
        that maximizes solution fitness"
        :param current_solution: The current solution to be optimised
        :return: A child solution
        """
        return current_solution