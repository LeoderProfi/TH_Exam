import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import scipy.sparse.csgraph
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix

class Problem():
    def __init__(self, n_elements, dimension):
        self.n_elements = n_elements
        self.dimension = dimension
        self.matrix, self.rhs = self.build_matrix_and_rhs(n_elements, dimension)
        self.reordered_matrix = self.reorder()
        #self.cholesky_factor, self.cholesky_time = self.cholesky(self.matrix)
        #self.cholesky_factor_reorderd, self.cholesky_time_reordered = self.cholesky(self.reordered_matrix)

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

    def cholesky(self, matrix):
        time_start = time.perf_counter()
        matrix_dense = matrix.toarray()
        cholesky_factor = np.linalg.cholesky(matrix_dense)
        time_taken = time.perf_counter() - time_start
        return cholesky_factor, time_taken

    def forward_substitution(self, L, b):
        """Perform forward substitution to solve Ly = b."""
        n = len(b)
        y = np.zeros_like(b)
        for i in range(n):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        return y

    def backward_substitution(self, L_T, y):
        """Perform backward substitution to solve L^T x = y."""
        n = len(y)
        x = np.zeros_like(y)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(L_T[i, i + 1:], x[i + 1:])) / L_T[i, i]
        return x

    def solve_cholesky(self, cholesky_factor):
        """Solve Ax = b using the Cholesky decomposition."""
        time_start = time.perf_counter()
        y = self.forward_substitution(cholesky_factor, self.rhs)
        time_forward_solve = time.perf_counter() - time_start

        time_start = time.perf_counter()
        x = self.backward_substitution(cholesky_factor.T, y)
        time_backward_solve = time.perf_counter() - time_start

        return x, time_forward_solve, time_backward_solve

    def fill_in_ratio(self, reorder=False):
        if reorder:
            cholesky_factor, _ = self.cholesky(self.reordered_matrix)
            nnz_A = self.reordered_matrix.nnz
            nnz_C = np.count_nonzero(cholesky_factor)
        else:
            cholesky_factor, _ = self.cholesky(self.matrix)
            nnz_A = self.matrix.nnz
            nnz_C = np.count_nonzero(cholesky_factor)

        return nnz_C / nnz_A

    def reorder(self):
        rcm_order = scipy.sparse.csgraph.reverse_cuthill_mckee(self.matrix)
        return self.matrix[rcm_order, :][:, rcm_order]

    def ssor_step(self,A,rhs, length_rhs, x, omega):
        # Forward sweep
        for i in range(length_rhs):
            sigma = x[i].copy()
            x[i] = (rhs[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
            x[i] = (1 - omega) * sigma + omega * x[i]

        # Backward sweep
        for i in reversed(range(length_rhs)):
            sigma = x[i].copy()
            x[i] = (rhs[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
            x[i] = (1 - omega) * sigma + omega * x[i]
        
        return x

    def ssor(self, x0=None, omega=1.5, tol=1e-10, max_iter=1000):
        n = len(self.rhs)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()

        A = self.matrix.todense()

        residuals = [np.linalg.norm(self.rhs - A @ x) / np.linalg.norm(self.rhs)]
        for k in range(max_iter):

            x = self.ssor_step(A, self.rhs, n, x, omega)

            residual = np.linalg.norm(self.rhs - A @ x) / np.linalg.norm(self.rhs)
            residuals.append(residual)
            if residual < tol:
                break

        return x, residuals

    def cg_ssor_step(self, Diagonal, Diag_omegaE, Diag_omegaF, x, residual, direction, omega = 1.5):
        
        #M_SSOR = Diag_omegaE @ np.diag(1 / np.diag(Diagonal)) @ Diag_omegaF

        #Forward solve
        y = self.forward_substitution(Diag_omegaE, residual)
        #Diagonal scaling
        w = y / np.diag(Diagonal)
        #Backward solve

        z = self.backward_substitution(Diag_omegaF, w)

        #Compute step size
        helper = np.dot(direction, self.matrix @ direction)

        if helper == 0:
            alpha = 1
        else:
            alpha = np.dot(residual.T, z) / helper

        #Update x
        x = x + alpha * direction

        #Update residual
        residual_new = residual - alpha * self.matrix @ direction

        #Forward solve
        y = self.forward_substitution(Diag_omegaE, residual_new)
        #Diagonal scaling
        w = y / np.diag(Diagonal)
        #Backward solve
        z_new = self.backward_substitution(Diag_omegaF, w)

        #Compute beta
        beta = np.dot(residual_new.T, z_new) / np.dot(residual.T, z)

        #Update direction
        direction = z_new + beta * direction


        return x, residual_new, direction

    def cg_with_ssor(self, tol=1e-10, max_iter=1000, omega =1.5):

        Residuals = []

        Diagonal = np.diag(np.diag(self.matrix.todense()))
        E = -np.tril(self.matrix.todense(), -1)
        F = -np.triu(self.matrix.todense(), 1)

        Residuals.append(np.linalg.norm(self.rhs - self.matrix @ np.zeros(len(self.rhs))) / np.linalg.norm(self.rhs))

        x = np.zeros_like(self.rhs)
        r = self.rhs - self.matrix @ x  
        
        y = self.forward_substitution((Diagonal + omega*E), r)
        w = y / np.diag(Diagonal)
        z = self.backward_substitution((Diagonal + omega*F).T, w)

        direction = z.copy()
        Residuals = [np.linalg.norm(r)/np.linalg.norm(self.rhs)]


        #x, r, direction = self.cg_ssor_step(Diagonal, E, F, np.zeros(len(self.rhs)), self.rhs/np.linalg.norm(self.rhs), np.zeros(len(self.rhs)))
        Diag_omegaE = np.diag(np.diag(self.matrix.todense())) + omega * np.tril(self.matrix.todense(), -1)
        Diag_omegaF = np.diag(np.diag(self.matrix.todense())) + omega * np.triu(self.matrix.todense(), 1)

        for i in range(max_iter):
            x, r, direction = self.cg_ssor_step(Diagonal, Diag_omegaE, Diag_omegaF, x, r, direction)
            Residuals.append(np.linalg.norm(r) / np.linalg.norm(self.rhs))
            if np.linalg.norm(r) < tol:
                break
        
        return x, Residuals

def compare_methods():
    p_values = range(2, 11)
    for p in p_values:
        n_elements = 2 * p - 1
        problem = Problem(n_elements, 3)
        u_exact = problem.exact_solution(n_elements)
        cho_fac, _ = problem.cholesky(problem.matrix)
        u_h_cholesky = problem.solve_cholesky(cho_fac)[0]
        u_h_ssor,_ = problem.ssor()
        u_h_cg, _ = problem.cg_with_ssor()
        u_h_spsolve = spsolve(problem.matrix, problem.rhs)
        error_cholesky = np.max(np.abs(u_h_cholesky - u_exact))
        error_spsolve = np.max(np.abs(u_h_spsolve - u_exact))
        error_ssor = np.max(np.abs(u_h_ssor- u_exact))
        error_cg = np.max(np.abs(u_h_cg - u_exact))
        print(f"p = {p}, Error (Cholesky): {error_cholesky}, Error (spsolve): {error_spsolve}, Error (ssor): {error_ssor}, Error (CG): {error_cg}")

def check_SPD(p=5):
    n_elements = 2 * p - 1
    problem = Problem(n_elements, 3)
    matrix = problem.matrix
    matrix_dense = matrix.toarray()

    symmetry_error = np.linalg.norm(matrix_dense - matrix_dense.T)
    print("Symmetry error:", symmetry_error)

    eigenvalues = np.linalg.eigvals(matrix_dense)
    is_spd = np.all(eigenvalues > 0)
    print("Is the matrix SPD?", is_spd)

def E2(dimension=2):
    if dimension == 2:
        p_values = range(2, 11)
    elif dimension == 3:
        p_values = range(2, 9)
    else:
        print("Wrong Dimension")
        return
    errors = []
    hs = []
    for p in p_values:
        print(f"p = {p}")
        n_elements = 2 * p - 1  
        h = 1.0 / (n_elements + 1) 
        problem = Problem(n_elements, dimension)
        u_h = spsolve(problem.matrix, problem.rhs)
        u_exact = problem.exact_solution(n_elements)
        error = np.max(np.abs(u_h - u_exact))
        errors.append(error)
        hs.append(h)

    plt.loglog(hs, errors, '-o', label="Error")
    scaled_h2 = [(h**2) * (errors[-1] / hs[-1]**2) for h in hs]
    plt.loglog(hs, scaled_h2, '--', label="h^2 (2nd order)")
    plt.gca().invert_xaxis()

    plt.xlabel("h (Grid Spacing)")
    plt.ylabel("Max Norm of Error")
    plt.legend()
    plt.savefig(f"E2_p_{p_values[-1]}_{dimension}.png")
    plt.clf()

def E3(dimension=2, reorder=False):
    if dimension == 2:
        p_values = range(2, 11)
    elif dimension == 3:
        p_values = range(2, 9)
    else:
        print("Wrong Dimension")
        return
    cholesky_times = []
    backward_solve_times = []
    forward_solve_times = []
    n = []
    for p in p_values:
        n_elements = 2 * p - 1
        n.append(n_elements**dimension)
        problem = Problem(n_elements, 3)
        if reorder:
            problem.matrix = problem.reordered_matrix
        chol_fac, cholesky_time = problem.cholesky(problem.matrix)
        cholesky_times.append(cholesky_time)
        _, forward_solve_time, backward_solve_time = problem.solve_cholesky(chol_fac)
        forward_solve_times.append(forward_solve_time)
        backward_solve_times.append(backward_solve_time)


    theoretical_times = [(n_i**3 + n_i)*(cholesky_times[0]/(n[0]**3 + n[0])) for n_i in n]

    # Plot the results
    plt.semilogy(n, theoretical_times, '--', label="Theoretical")
    plt.semilogy(n, cholesky_times, '-o', label="Cholesky (Measured)")
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if reorder:
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}_Cholesky.png")
    else:
        plt.savefig(f"3/E3_p_{p_values[-1]}_{dimension}_Cholesky.png")
    plt.clf()

    theoretical_times = [(n_i**2 + n_i)*(forward_solve_times[0]/(n[0]**2 + n[0])) for n_i in n]

    # Plot the results
    plt.semilogy(n, theoretical_times, '--', label="Theoretical")
    plt.semilogy(n, forward_solve_times, '-o', label="forward solve (Measured)")
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if reorder:
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}_Forward.png")
    else:
        plt.savefig(f"3/E3_p_{p_values[-1]}_{dimension}_Forward.png")
    plt.clf()

    theoretical_times = [(n_i**2 + n_i)*(backward_solve_times[0]/(n[0]**2 + n[0])) for n_i in n]

    # Plot the results
    plt.semilogy(n, theoretical_times, '--', label="Theoretical")
    plt.semilogy(n, backward_solve_times, '-o', label="backward solve (Measured)")
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if reorder:
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}_Backward.png")
    else:
        plt.savefig(f"3/E3_p_{p_values[-1]}_{dimension}_Backward.png")
    plt.clf()




    plt.semilogy(n, cholesky_times, '-o', label="Cholesky")
    plt.semilogy(n, forward_solve_times, '-o', label="Forward Solve")
    plt.semilogy(n, backward_solve_times, '-o', label="Backward Solve")
    plt.xlabel("N (Number of Elements)")
    plt.ylabel("Time (s)")
    plt.legend()
    if reorder:
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}.png")
    else:
        plt.savefig(f"3/E3_p_{p_values[-1]}_{dimension}.png")
    plt.clf()

def E4(dimension=2, reorder=False):
    if dimension == 2:
        p_values = range(2, 30)
    elif dimension == 3:
        p_values = range(2, 15)
    else:
        print("Wrong Dimension")
        return
    problem_size = []
    fill_in_ratios = []
    for p in p_values:
        n_elements = 2 * p - 1
        
        problem = Problem(n_elements, dimension)
        fill_in_ratio = problem.fill_in_ratio(reorder)
        problem_size.append(n_elements**dimension)
        fill_in_ratios.append(fill_in_ratio)

    if dimension == 2:
        def fill_in(n, a, b, c):
            return a * n**0.5 + b * n + c
    elif dimension == 3:
        def fill_in(n, a, b, c):
            return a * n**(1/3) + b * n + c
    
    
    popt, _ = curve_fit(fill_in, problem_size, fill_in_ratios)
    print("a, b, c:", popt)

    plt.plot(problem_size, [fill_in(n, *popt) for n in problem_size], '--', label="Fit")
    plt.plot(problem_size, fill_in_ratios, '-o')
    plt.legend(["Fit", "Measured"])
    plt.xlabel("Problem Size N (Number of Elements)")
    plt.ylabel("Fill-in Ratio")
    if reorder:
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}.png")
    else:
        plt.savefig(f"4/E4_p_{p_values[-1]}_{dimension}.png")
    plt.clf()

def E5(dimension=2, a=False, b=False):
    if a:
        
        if dimension == 2:
            p_values = range(2, 11)
        elif dimension == 3:
            p_values = range(2, 9)
        else:
            print("Wrong Dimension")
            return
        cholesky_times = []
        backward_solve_times = []
        forward_solve_times = []

        cholesky_times_reorderd = []
        backward_solve_times_reorderd = []
        forward_solve_times_reorderd = []

        n = []
        for p in p_values:
            n_elements = 2 * p - 1
            n.append(n_elements**dimension)
            problem = Problem(n_elements, 3)

            chol_fac, cholesky_time = problem.cholesky(problem.matrix)
            _, forward_solve_time, backward_solve_time = problem.solve_cholesky(chol_fac)
            forward_solve_times.append(forward_solve_time)
            backward_solve_times.append(backward_solve_time)
            cholesky_times.append(cholesky_time)
            
            
            chol_fac_r, cholesky_time_reo = problem.cholesky(problem.reordered_matrix)
            _, forward_solve_time_reo, backward_solve_time_reo = problem.solve_cholesky(chol_fac_r)
            forward_solve_times_reorderd.append(forward_solve_time_reo)
            backward_solve_times_reorderd.append(backward_solve_time_reo)
            cholesky_times_reorderd.append(cholesky_time_reo)
        
        theoretical_times_chol = [(n_i**3 + n_i)*(cholesky_times[0]/(n[0]**3 + n[0])) for n_i in n]

        # Plot the results
        plt.semilogy(n, theoretical_times_chol, '--', label="Theoretical")
        plt.semilogy(n, cholesky_times, '-o', label="Cholesky (Measured)")
        plt.semilogy(n, cholesky_times_reorderd, '-o', label="Cholesky (Measured) Reordered")
        plt.xlabel("Problem Size (N)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}_Cholesky.png")
        plt.clf()

        theoretical_times = [(n_i**2 + n_i)*(forward_solve_times[0]/(n[0]**2 + n[0])) for n_i in n]

        # Plot the results
        plt.semilogy(n, theoretical_times, '--', label="Theoretical")
        plt.semilogy(n, forward_solve_times, '-o', label="forward solve (Measured)")
        plt.semilogy(n, forward_solve_times_reorderd, '-o', label="forward solve (Measured) Reordered")
        plt.xlabel("Problem Size (N)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}_Forward.png")
        plt.clf()

        theoretical_times = [(n_i**2 + n_i)*(backward_solve_times[0]/(n[0]**2 + n[0])) for n_i in n]

        # Plot the results
        plt.semilogy(n, theoretical_times, '--', label="Theoretical")
        plt.semilogy(n, backward_solve_times, '-o', label="backward solve (Measured)")
        plt.semilogy(n, backward_solve_times_reorderd, '-o', label="backward solve (Measured) Reordered")
        plt.xlabel("Problem Size (N)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}_Backward.png")
        plt.clf()


        plt.semilogy(n, cholesky_times_reorderd, '-o', label="Cholesky reorderd")
        plt.semilogy(n, forward_solve_times_reorderd, '-o', label="Forward Solve reorderd")
        plt.semilogy(n, backward_solve_times_reorderd, '-o', label="Backward Solve reorderd")
        plt.xlabel("N (Number of Elements)")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.savefig(f"5/E5_p_{p_values[-1]}_{dimension}.png")
        plt.clf()


    #################################
    # Fill in ratio
    #################################
    if b:
        if dimension == 2:
            p_values = range(2, 11)
        elif dimension == 3:
            p_values = range(2, 9)
        else:
            print("Wrong Dimension")
            return
        
        problem_size = []
        fill_in_ratios = []
        fill_in_ratios_reordered = []
        for p in p_values:
            n_elements = 2 * p - 1
            
            problem = Problem(n_elements, dimension)
            fill_in_ratio = problem.fill_in_ratio()
            fill_in_ratio_reordered = problem.fill_in_ratio(True)
            problem_size.append(n_elements**dimension)
            fill_in_ratios.append(fill_in_ratio)
            fill_in_ratios_reordered.append(fill_in_ratio_reordered)



        plt.plot(problem_size, fill_in_ratios, '-o')
        plt.plot(problem_size, fill_in_ratios_reordered, '-o')
        plt.legend(["Original", "Reordered"])
        plt.xlabel("Problem Size N (Number of Elements)")
        plt.ylabel("Fill-in Ratio")
        plt.savefig(f"5/fillIn_p_{p_values[-1]}_{dimension}.png")
        plt.clf()

def E6():
    p_values = range(6, 11)
    for p in p_values:
        n_elements = 2 * p - 1
        problem = Problem(n_elements, 2)
        u_h, residuals = Problem.ssor(problem)
        plt.semilogy(residuals, label=f"p = {p}")

    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.legend()
    plt.savefig("6/E6.png")
    plt.clf()

def E7():
    p_values = range(6, 11)
    as_rate = []
    for p in p_values:
        n_elements = 2 * p - 1
        problem = Problem(n_elements, 2)
        u_h, residuals = Problem.ssor(problem)
        as_rate.append(residuals[-5:])
    print(np.array(as_rate).T)
           
def E8(dim=2):
    if dim == 2:
        p_values = range(6, 10)
    elif dim == 3:
        p_values = range(6, 15)
    else:
        print("Invalid dimension")
    direct_times = []
    ssor_times = []
    N_vals = []
    for p in p_values:
        print(f"p = {p}")
        n_elements = 2 * p - 1
        N_vals = np.append(N_vals, n_elements**dim)
        prob = Problem(n_elements, dim)
        
        # Time for direct solve
        time_start = time.perf_counter()
        direct_sol = spsolve(prob.matrix, prob.rhs)
        time_direct_sol = time.perf_counter() - time_start

        # Time for SSOR solve
        time_start = time.perf_counter()
        ssor_sol, ssor_res = prob.ssor()
        time_ssor = time.perf_counter() - time_start

        direct_times.append(time_direct_sol)
        ssor_times.append(time_ssor)

    #theo_time = [(ssor_times[0]/(len(ssor_res)*(2 * prob.matrix.nnz * N_vals[0] + N_vals[0])))*(len(ssor_res)*(2 * prob.matrix.nnz * n_v + n_v)) for n_v in N_vals]
    theo_time = [(ssor_times[0]/(len(ssor_res)*(2 * N_vals[0]**2 + N_vals[0])))*(len(ssor_res)*(2 * n_v**2 + n_v)) for n_v in N_vals]

    plt.figure(figsize=(10, 6))
    plt.semilogy(N_vals, direct_times, 'o-', label="Direct Solve", markersize=5, color='blue')
    plt.semilogy(N_vals, ssor_times, 's-', label="SSOR", markersize=5, color='green')
    plt.semilogy(N_vals, theo_time, 'd--', label="Theoretical Time", markersize=5, color='red')

    plt.xlabel("N (Number of Elements)")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Direct Solve and SSOR Times")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"8/E8_{dim}.png")

"""The theoretical computation time for the SSOR is calculatred through the FLOPs per iteration times the number of iterations. We have two matrix vector products and two vector updates per iteration, resulting in a theoretical computation time of 2 * n*2 for the matrix * vector, (2cn, with c nnz for sparse matrices.) + 2n for the vector alloc (to fact check) """



def E9():
    p_values = range(8, 15)
    plt.figure(figsize=(10, 6))  # Set the figure size

    for p in p_values:
        n_elements = 2 * p - 1
        problem = Problem(n_elements, 2)
        u_h, residuals = Problem.ssor(problem)
        u_cg, residuals_cg = Problem.cg_with_ssor(problem)

        plt.semilogy(residuals, label=f"SSOR, p = {p}", linestyle='-', marker='o', markersize=3, markevery=10)
        plt.semilogy(residuals_cg, label=f"CG with SSOR, p = {p}", linestyle='--', marker='x', markersize=3, markevery=10)

    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Convergence of SSOR and CG with SSOR")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("9/E9.png")
    plt.clf()

def E10(dim=2):
    if dim == 2:
        p_values = range(6, 10)
    elif dim == 3:
        p_values = range(6, 15)
    else:
        print("Invalid dimension")
    direct_times = []
    ssor_times = []
    cg_times = []
    N_vals = []
    for p in p_values:
        print(f"p = {p}")
        n_elements = 2 * p - 1
        N_vals = np.append(N_vals, n_elements**dim)
        prob = Problem(n_elements, dim)
        
        # Time for direct solve
        time_start = time.perf_counter()
        direct_sol = spsolve(prob.matrix, prob.rhs)
        time_direct_sol = time.perf_counter() - time_start

        # Time for SSOR solve
        time_start = time.perf_counter()
        ssor_sol, ssor_res = prob.ssor()
        time_ssor = time.perf_counter() - time_start

        # Time for CG solve
        time_start = time.perf_counter()
        cg_sol = prob.cg_with_ssor()
        time_cg = time.perf_counter() - time_start
    

        direct_times.append(time_direct_sol)
        ssor_times.append(time_ssor)
        cg_times.append(time_cg)

    #theo_time = [(ssor_times[0]/(len(ssor_res)*(2 * prob.matrix.nnz * N_vals[0] + N_vals[0])))*(len(ssor_res)*(2 * prob.matrix.nnz * n_v + n_v)) for n_v in N_vals]

    plt.figure(figsize=(10, 6))
    plt.semilogy(N_vals, direct_times, 'o-', label="Direct Solve", markersize=5, color='blue')
    plt.semilogy(N_vals, ssor_times, 's-', label="SSOR", markersize=5, color='green')
    plt.semilogy(N_vals, cg_times, 'd-', label="CG with SSOR", markersize=5, color='red')

    plt.xlabel("N (Number of Elements)")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Direct Solve and SSOR Times")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"10/E10_{dim}.png")



#compare_methods()

# E2(2)
# E2(3)

# E3(2)
# E3(3)

# E4(2)
# E4(3)

# E3(2, reorder=True)
# E3(3, reorder=True)

# E4(2, reorder=True)
# E4(3, reorder=True)

# E5(2, a=True, b=True)
# E5(3, a=True, b=True)
# E6()

#E7()
#E8(2)
E9()
#E10(2)


#compare_methods()

#p = Problem(10,2)

# plt.spy(p.matrix, markersize=0.5)
# plt.savefig("spaity_pattern_matrix.png")
# plt.clf()

# plt.spy(p.cholesky_factor, markersize=0.5)
# plt.savefig("spaity_pattern_cholesky.png")
# plt.clf()

# plt.spy(p.reordered_matrix, markersize=0.5)
# plt.savefig("spaity_pattern_reordered.png")
# plt.clf()

# plt.spy(p.cholesky_factor_reorderd, markersize=0.5)
# plt.savefig("spaity_pattern_cholesky_reordered.png")
# plt.clf()