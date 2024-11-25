import numpy as np


class Problem:
    def __init__(self):
        self.cost_matrix = [[]]

    def generate_problem(self, num_tasks):
        """
        Randomly generates a problem of size `number_tasks`
        :return: Randomly generated symmetrical cost matrix representing the problem
        """
        # Set the random seed for reproducibility
        np.random.seed(0)
        # Generate a random symmetrical 20x20 cost matrix
        size = num_tasks
        cost_matrix = np.random.randint(1, 100, size=(size, size))
        # Make the matrix symmetrical
        cost_matrix = (cost_matrix + cost_matrix.T) // 2
        # Set the diagonal to zero (no cost for staying at the same location)
        np.fill_diagonal(cost_matrix, 0)
        self.cost_matrix = cost_matrix

    def save_cost_matrix(self, filename, format='csv'):
        """
        Saves the cost matrix to a file in the specified format.
        :param filename: Name of the file to save.
        :param format: Format to save the file ('csv' or 'npy').
        """
        if format == 'csv':
            np.savetxt(filename, self.cost_matrix, delimiter=',', fmt='%d')
        elif format == 'npy':
            np.save(filename, self.cost_matrix)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'npy'.")

    def load_cost_matrix(self, filename, format='csv'):
        """
        Loads the cost matrix from a file in the specified format.
        :param filename: Name of the file to load.
        :param format: Format of the file ('csv' or 'npy').
        """
        if format == 'csv':
            self.cost_matrix = np.loadtxt(filename, delimiter=',', dtype=int)
        elif format == 'npy':
            self.cost_matrix = np.load(filename)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'npy'.")
