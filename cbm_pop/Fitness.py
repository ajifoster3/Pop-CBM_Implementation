
class Fitness:

    @staticmethod
    def fitness_function(solution, cost_matrix):
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