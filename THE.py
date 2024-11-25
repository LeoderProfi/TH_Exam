import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class Problem():
    def __init__(self, n_elements, dimension):
        self.n_elements = n_elements
        self.dimension = dimension
        self.matrix, self.rhs = self.build_matrix_and_rhs(n_elements, dimension)
        self.cholesky_factor, self.cholesky_time = self.cholesky()

    def build_matrix_and_rhs(self, n_elements, dimension):
        spacing = 1.0 / (n_elements + 1)
        if dimension == 2:
            num_points = n_elements * n_elements
            A = lil_matrix((num_points, num_points))
            f_vector = np.zeros(num_points)
            boundary_nodes = []

            # Define the source function f(x, y)
            def f(x, y):
                return (x**2 + y**2) * math.sin(x * y)

            for i in range(n_elements):
                for j in range(n_elements):
                    p = i * n_elements + j
                    x = (i + 1) * spacing
                    y = (j + 1) * spacing

                    # Check if the point is on the boundary
                    if i == 0 or i == n_elements - 1 or j == 0 or j == n_elements - 1:
                        # Boundary condition: u = u_0
                        u_0 = math.sin(x * y)
                        A[p, p] = 1
                        f_vector[p] = u_0
                        boundary_nodes.append(p)
                    else:
                        # Interior points
                        A[p, p] = 4
                        neighbors = self.get_neighbors_2d(p, n_elements)
                        for neighbor_p in neighbors:
                            A[p, neighbor_p] = -1
                        f_vector[p] = f(x, y) * spacing**2

            # Adjust RHS for interior nodes to account for boundary conditions
            for p in boundary_nodes:
                u_0 = f_vector[p]  # Boundary value
                neighbors = self.get_neighbors_2d(p, n_elements)
                for q in neighbors:
                    # Only adjust if q is an interior node
                    if A[q, p] != 0:
                        f_vector[q] -= A[q, p] * u_0

            # Zero out columns and rows for boundary nodes to maintain symmetry
            for p in boundary_nodes:
                A[p, :] = 0
                A[:, p] = 0
                A[p, p] = 1

            return A.tocsr(), f_vector

        elif dimension == 3:
            num_points = n_elements * n_elements * n_elements
            A = lil_matrix((num_points, num_points))
            f_vector = np.zeros(num_points)
            boundary_nodes = []

            # Define the source function f(x, y, z)
            def f(x, y, z):
                return (x**2 * y**2 + z**2 * y**2 + x**2 * z**2) * math.sin(x * y * z)

            for i in range(n_elements):
                for j in range(n_elements):
                    for k in range(n_elements):
                        p = i * n_elements * n_elements + j * n_elements + k
                        x = (i + 1) * spacing
                        y = (j + 1) * spacing
                        z = (k + 1) * spacing

                        # Check if the point is on the boundary
                        if (i == 0 or i == n_elements - 1 or
                            j == 0 or j == n_elements - 1 or
                            k == 0 or k == n_elements - 1):
                            # Boundary condition: u = u_0
                            u_0 = math.sin(x * y * z)
                            A[p, p] = 1
                            f_vector[p] = u_0
                            boundary_nodes.append(p)
                        else:
                            # Interior points
                            A[p, p] = 6
                            neighbors = self.get_neighbors_3d(p, n_elements)
                            for neighbor_p in neighbors:
                                A[p, neighbor_p] = -1
                            f_vector[p] = f(x, y, z) * spacing**2

            # Adjust RHS for interior nodes to account for boundary conditions
            for p in boundary_nodes:
                u_0 = f_vector[p]  # Boundary value
                neighbors = self.get_neighbors_3d(p, n_elements)
                for q in neighbors:
                    # Only adjust if q is an interior node
                    if A[q, p] != 0:
                        f_vector[q] -= A[q, p] * u_0

            # Zero out columns and rows for boundary nodes to maintain symmetry
            for p in boundary_nodes:
                A[p, :] = 0
                A[:, p] = 0
                A[p, p] = 1

            return A.tocsr(), f_vector
        else:
            print("Invalid dimension")
            return None, None

    def get_neighbors_2d(self, p, n_elements):
        i = p // n_elements
        j = p % n_elements
        neighbors = []
        if j > 0:
            neighbors.append(p - 1)  # Left neighbor
        if j < n_elements - 1:
            neighbors.append(p + 1)  # Right neighbor
        if i > 0:
            neighbors.append(p - n_elements)  # Bottom neighbor
        if i < n_elements - 1:
            neighbors.append(p + n_elements)  # Top neighbor
        return neighbors

    def get_neighbors_3d(self, p, n_elements):
        n2 = n_elements * n_elements
        i = p // n2
        j = (p % n2) // n_elements
        k = p % n_elements
        neighbors = []
        if k > 0:
            neighbors.append(p - 1)  # Back neighbor
        if k < n_elements - 1:
            neighbors.append(p + 1)  # Front neighbor
        if j > 0:
            neighbors.append(p - n_elements)  # Left neighbor
        if j < n_elements - 1:
            neighbors.append(p + n_elements)  # Right neighbor
        if i > 0:
            neighbors.append(p - n2)  # Bottom neighbor
        if i < n_elements - 1:
            neighbors.append(p + n2)  # Top neighbor
        return neighbors

    def exact_solution(self, n_elements):
        spacing = 1.0 / (n_elements + 1)
        if self.dimension == 2:
            u_exact = np.zeros(n_elements * n_elements)
            for i in range(n_elements):
                for j in range(n_elements):
                    x = (i + 1) * spacing
                    y = (j + 1) * spacing
                    p = i * n_elements + j
                    u_exact[p] = math.sin(x * y)
            return u_exact
        elif self.dimension == 3:
            u_exact = np.zeros(n_elements * n_elements * n_elements)
            for i in range(n_elements):
                for j in range(n_elements):
                    for k in range(n_elements):
                        x = (i + 1) * spacing
                        y = (j + 1) * spacing
                        z = (k + 1) * spacing
                        p = i * n_elements * n_elements + j * n_elements + k
                        u_exact[p] = math.sin(x * y * z)
            return u_exact
        else:
            print("Invalid dimension")
            return None

    def cholesky(self):
        time_start = time.perf_counter()
        matrix_dense = self.matrix.toarray()
        cholesky_factor = np.linalg.cholesky(matrix_dense)
        time_taken = time.perf_counter() - time_start
        return cholesky_factor, time_taken


    def solve_cholesky(self):
        # Forward substitution: Solve L y = b
        time_start = time.time()
        y = np.linalg.solve(self.cholesky_factor, self.rhs)
        time_forward_solve = time.time() - time_start

        # Backward substitution: Solve L^T x = y
        time_start = time.time()
        x = np.linalg.solve(self.cholesky_factor.T, y)
        time_backward_solve = time.time() - time_start

        return x, time_forward_solve, time_backward_solve

def verify_second_order_accuracy():
    p_values = range(2, 6)
    errors = []
    hs = []
    for p in p_values:
        n_elements = 2 * p - 1  # Ensure n_elements is an integer
        h = 1.0 / (n_elements + 1)  # Compute spacing based on n_elements
        problem = Problem(n_elements, 2)  # Change to 2D for quicker computation
        u_h = spsolve(problem.matrix, problem.rhs)
        u_exact = problem.exact_solution(n_elements)
        error = np.max(np.abs(u_h - u_exact))
        errors.append(error)
        hs.append(h)
        print(f"h = {h}, error = {error}")

    plt.loglog(hs, errors, '-o', label="Error")
    plt.loglog(hs, [h**2 for h in hs], '--', label="h^2 (2nd order)")
    plt.xlabel("h (Grid Spacing)")
    plt.ylabel("Max Norm of Error")
    plt.legend()
    plt.title("Verification of Second-Order Accuracy")
    plt.show()

def cholesky_timing():
    p_values = range(2, 11)
    cholesky_times = []
    backward_solve_times = []
    forward_solve_times = []
    for p in p_values:
        n_elements = 2 * p - 1  # Ensure n_elements is an integer
        problem = Problem(n_elements, 3)  # Change to 2D for quicker computation
        _, cholesky_time = problem.cholesky()
        cholesky_times.append(cholesky_time)
        _, forward_solve_time, backward_solve_time = problem.solve_cholesky()
        forward_solve_times.append(forward_solve_time)
        backward_solve_times.append(backward_solve_time)
        print(f"p = {p}, Cholesky time = {cholesky_time}")

    plt.loglog(p_values, cholesky_times, '-o', label="Cholesky")
    plt.loglog(p_values, forward_solve_times, '-o', label="Forward Solve")
    plt.loglog(p_values, backward_solve_times, '-o', label="Backward Solve")
    plt.xlabel("p (Number of Elements)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title("Timing Analysis of Cholesky Decomposition and Solves")
    plt.show()

def compare_methods():
    p_values = range(2, 11)
    for p in p_values:
        n_elements = 2 * p - 1
        problem = Problem(n_elements, 3)  # Change to 2D for quicker computation
        u_exact = problem.exact_solution(n_elements)
        u_h_cholesky = problem.solve_cholesky()[0]
        u_h_spsolve = spsolve(problem.matrix, problem.rhs)
        error_cholesky = np.max(np.abs(u_h_cholesky - u_exact))
        error_spsolve = np.max(np.abs(u_h_spsolve - u_exact))
        print(f"p = {p}, Error (Cholesky): {error_cholesky}, Error (spsolve): {error_spsolve}")

def check_SPD(p=5):
    n_elements = 2 * p - 1
    problem = Problem(n_elements, 3)  # Change to 2D for quicker computation
    matrix = problem.matrix
    matrix_dense = matrix.toarray()

    # Check symmetry
    symmetry_error = np.linalg.norm(matrix_dense - matrix_dense.T)
    print("Symmetry error:", symmetry_error)  # Should be close to zero

    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(matrix_dense)
    is_spd = np.all(eigenvalues > 0)
    print("Is the matrix SPD?", is_spd)

# Run the verification
#verify_second_order_accuracy()

# Run the timing analysis
cholesky_timing()

# Run the comparison of methods
compare_methods()

# Run the check for SPD
check_SPD(5)
