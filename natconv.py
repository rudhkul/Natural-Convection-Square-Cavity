import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

class NaturalConvectionSolverFV:
    def __init__(
        self, 
        nx: int = 101, 
        ny: int = 101, 
        Ra: float = 1e6, 
        Pr: float = 0.71,
        total_time: float = 1,
        cfl: float = 0.95,
        dt_fixed: Optional[float] = None,  # Manual time step parameter
        piso_iterations: int = 2,         # Number of PISO iterations
        relaxation_factors: dict = {
            "U": 0.7,
            "T": 0.7,
            "p_rgh": 0.3
        }
    ):
        # Domain parameters
        self.nx = nx
        self.ny = ny
        self.Lx = 1.0
        self.Ly = 1.0
        
        # Grid spacing
        self.dx = self.Lx / (nx - 1)
        self.dy = self.Ly / (ny - 1)
        
        # Non-dimensional parameters
        self.Ra = Ra  # Rayleigh number
        self.Pr = Pr  # Prandtl number
        self.cfl = cfl  # CFL number
        
        # Time parameters
        self.total_time = total_time
        self.dt = 0.0
        self.dt_fixed = dt_fixed  # Manual time step control
        
        # Solution arrays
        self.u = np.zeros((ny, nx), dtype=np.float64)  # Velocity in x-direction
        self.v = np.zeros((ny, nx), dtype=np.float64)  # Velocity in y-direction
        self.p = np.zeros((ny, nx), dtype=np.float64)  # Pressure
        self.T = np.zeros((ny, nx), dtype=np.float64)  # Temperature
        
        # Temporary storage arrays
        self.u_old = np.zeros_like(self.u)
        self.v_old = np.zeros_like(self.v)
        self.T_old = np.zeros_like(self.T)
        
        # Grid coordinates
        self.x = np.linspace(0, self.Lx, nx)
        self.y = np.linspace(0, self.Ly, ny)
        
        # Physical parameters
        self.g = 1.0  # Gravitational acceleration
        self.beta = 1.0  # Thermal expansion coefficient
        
        # Numerical parameters
        self.max_iter = 1000
        self.tolerance = 1e-4
        self.piso_iterations = piso_iterations  # Number of PISO iterations
        self.relaxation_factors = relaxation_factors  # Relaxation factors
        
        # Initialize fields and precompute matrices
        self._initialize_fields()
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy
        self.inv_dx2 = self.inv_dx ** 2
        self.inv_dy2 = self.inv_dy ** 2
        self._precompute_pressure_solver()

    def _initialize_fields(self):
        """
        Set initial and boundary conditions for velocity, pressure, and temperature.
        """
        # Temperature boundary conditions
        self.T[:, 0] = 1.0   # Hot wall (left)
        self.T[:, -1] = 0.0  # Cold wall (right)
        self.T[0, :] = self.T[1, :]  # Insulated top
        self.T[-1, :] = self.T[-2, :]  # Insulated bottom
        
        # Velocity boundary conditions (no-slip)
        self.u[:, :] = 0.0
        self.v[:, :] = 0.0

    def flux_limiter(self, r: float) -> float:
        """
        Van Leer flux limiter for stabilization.
        """
        return (r + abs(r)) / (1 + abs(r))

    def _compute_diffusion(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute diffusion term using central differencing (second-order accurate).
        """
        diff = np.zeros_like(phi)
        phi_j_plus = phi[1:-1, 2:]
        phi_j = phi[1:-1, 1:-1]
        phi_j_minus = phi[1:-1, :-2]
        diff[1:-1, 1:-1] += (phi_j_plus - 2 * phi_j + phi_j_minus) * self.inv_dx2
        
        phi_i_plus = phi[2:, 1:-1]
        phi_i = phi[1:-1, 1:-1]
        phi_i_minus = phi[:-2, 1:-1]
        diff[1:-1, 1:-1] += (phi_i_plus - 2 * phi_i + phi_i_minus) * self.inv_dy2
        
        return diff

    def _compute_convection_limited(self, phi: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute convection term with linearUpwind scheme (second-order accurate with slope limiting).
        """
        conv = np.zeros_like(phi)
        i_slice = slice(1, -1)  # Interior indices i=1 to ny-2
        j_slice = slice(1, -1)  # Interior indices j=1 to nx-2
        
        # X-direction components
        phi_j_plus = phi[i_slice, 2:]
        phi_j = phi[i_slice, j_slice]
        phi_j_minus = phi[i_slice, :-2]
        grad_phi_x = (phi_j_plus - phi_j_minus) * 0.5 * self.inv_dx  # Gradient in x-direction
        phi_x_plus = phi_j + 0.5 * grad_phi_x * self.dx
        phi_x_minus = phi_j - 0.5 * grad_phi_x * self.dx
        
        # Y-direction components
        phi_i_plus = phi[2:, j_slice]
        phi_i = phi[i_slice, j_slice]
        phi_i_minus = phi[:-2, j_slice]
        grad_phi_y = (phi_i_plus - phi_i_minus) * 0.5 * self.inv_dy  # Gradient in y-direction
        phi_y_plus = phi_i + 0.5 * grad_phi_y * self.dy
        phi_y_minus = phi_i - 0.5 * grad_phi_y * self.dy
        
        # Combine components
        conv_x = u[i_slice, j_slice] * (phi_x_plus - phi_x_minus) * self.inv_dx
        conv_y = v[i_slice, j_slice] * (phi_y_plus - phi_y_minus) * self.inv_dy
        conv[i_slice, j_slice] = conv_x + conv_y
        
        return conv

    def solve_energy_equation(self):
        """
        Solve energy equation using second-order implicit time stepping (backward scheme).
        """
        # Store previous temperature
        self.T_old = self.T.copy()
        
        # Compute diffusion and convection terms
        T_diff = self._compute_diffusion(self.T_old)
        T_conv = self._compute_convection_limited(self.T_old, self.u, self.v)
        
        # Update temperature with relaxation
        omega_T = self.relaxation_factors["T"]
        self.T[1:-1, 1:-1] = (
            (1 - omega_T) * self.T[1:-1, 1:-1] +
            omega_T * np.clip(
                self.T_old[1:-1, 1:-1] + self.dt * (
                    (1 / self.Pr) * T_diff[1:-1, 1:-1] - 
                    T_conv[1:-1, 1:-1]
                ),
                0, 1
            )
        )
        
        # Enforce boundary conditions
        self.T[:, 0] = 1.0   # Hot wall
        self.T[:, -1] = 0.0  # Cold wall
        self.T[0, :] = self.T[1, :]
        self.T[-1, :] = self.T[-2, :]

    def solve_momentum_equations(self):
        """
        Solve momentum equations using second-order implicit time stepping (backward scheme).
        """
        # Compute buoyancy term
        T_ref = 0.5
        buoyancy = self.g * self.beta * (self.T - T_ref)
        
        # Store previous velocities
        self.u_old = self.u.copy()
        self.v_old = self.v.copy()
        
        # Compute diffusion terms
        u_diff = self._compute_diffusion(self.u_old)
        v_diff = self._compute_diffusion(self.v_old)
        
        # Compute convection terms with linearUpwind scheme
        u_conv = self._compute_convection_limited(self.u_old, self.u, self.v)
        v_conv = self._compute_convection_limited(self.v_old, self.u, self.v)
        
        # Update u-velocity with relaxation
        omega_U = self.relaxation_factors["U"]
        self.u[1:-1, 1:-1] = (
            (1 - omega_U) * self.u[1:-1, 1:-1] +
            omega_U * (
                self.u_old[1:-1, 1:-1] + self.dt * (
                    (1 / self.Pr) * u_diff[1:-1, 1:-1] -
                    u_conv[1:-1, 1:-1]
                )
            )
        )
        
        # Update v-velocity (include buoyancy) with relaxation
        self.v[1:-1, 1:-1] = (
            (1 - omega_U) * self.v[1:-1, 1:-1] +
            omega_U * np.clip(
                self.v_old[1:-1, 1:-1] + self.dt * (
                    (1 / self.Pr) * v_diff[1:-1, 1:-1] - 
                    v_conv[1:-1, 1:-1] - 
                    self.Ra * self.Pr * buoyancy[1:-1, 1:-1]
                ),
                -1e3, 1e3
            )
        )
        
        # Enforce velocity boundary conditions
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

    def _precompute_pressure_solver(self):
        """
        Precompute pressure Poisson matrix and its LU decomposition with zero gradient boundary conditions.
        """
        n = self.nx * self.ny
        A = sparse.lil_matrix((n, n))
        def idx(i, j): return i * self.nx + j

        # Interior points
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                current = idx(i, j)
                A[current, current] = -2 * (self.inv_dx2 + self.inv_dy2)
                A[current, idx(i, j-1)] = self.inv_dx2  # Left neighbor
                A[current, idx(i, j+1)] = self.inv_dx2  # Right neighbor
                A[current, idx(i-1, j)] = self.inv_dy2  # Bottom neighbor
                A[current, idx(i+1, j)] = self.inv_dy2  # Top neighbor

        # Zero gradient boundary conditions
        for i in range(self.ny):
            # Left wall
            A[idx(i, 0), idx(i, 0)] = 1.0
            A[idx(i, 0), idx(i, 1)] = -1.0
            # Right wall
            A[idx(i, self.nx-1), idx(i, self.nx-1)] = 1.0
            A[idx(i, self.nx-1), idx(i, self.nx-2)] = -1.0

        for j in range(self.nx):
            # Bottom wall
            A[idx(0, j), idx(0, j)] = 1.0
            A[idx(0, j), idx(1, j)] = -1.0
            # Top wall
            A[idx(self.ny-1, j), idx(self.ny-1, j)] = 1.0
            A[idx(self.ny-1, j), idx(self.ny-2, j)] = -1.0

        # Reference pressure point
        ref_i, ref_j = self.ny // 2, self.nx // 2
        A[idx(ref_i, ref_j), :] = 0
        A[idx(ref_i, ref_j), idx(ref_i, ref_j)] = 1.0

        # Store for reuse
        self.A_pressure = A.tocsr()
        self.lu_pressure = splinalg.splu(self.A_pressure.tocsc())

    def solve_piso_algorithm(self):
        """
        Solve pressure-velocity coupling using the PISO algorithm.
        """
        # Step 1: Predict velocity without pressure gradient
        u_star = self.u.copy()
        v_star = self.v.copy()
        
        # Compute intermediate velocity fields
        u_star[1:-1, 1:-1] = self.u_old[1:-1, 1:-1] + self.dt * (
            (1 / self.Pr) * self._compute_diffusion(self.u_old)[1:-1, 1:-1] -
            self._compute_convection_limited(self.u_old, self.u, self.v)[1:-1, 1:-1]
        )
        
        v_star[1:-1, 1:-1] = self.v_old[1:-1, 1:-1] + self.dt * (
            (1 / self.Pr) * self._compute_diffusion(self.v_old)[1:-1, 1:-1] -
            self._compute_convection_limited(self.v_old, self.u, self.v)[1:-1, 1:-1] -
            self.Ra * self.Pr * (self.g * self.beta * (self.T[1:-1, 1:-1] - 0.5))
        )
        
        # Step 2: PISO iterations
        for _ in range(self.piso_iterations):
            # Compute divergence of intermediate velocity field
            div_u_star = np.zeros_like(self.p)
            div_u_star[1:-1, 1:-1] = (
                (u_star[1:-1, 2:] - u_star[1:-1, 1:-1]) * self.inv_dx +
                (v_star[2:, 1:-1] - v_star[1:-1, 1:-1]) * self.inv_dy
            )
            
            # Build RHS vector for pressure correction
            b = np.zeros(self.ny * self.nx)
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    b[i*self.nx + j] = div_u_star[i, j] / self.dt
            
            # Set reference pressure
            ref_idx = (self.ny // 2) * self.nx + (self.nx // 2)
            b[ref_idx] = 0.0
            
            # Solve pressure correction equation
            p_corr = self.lu_pressure.solve(b).reshape(self.ny, self.nx)
            
            # Correct velocity fields with relaxation
            omega_p = self.relaxation_factors["p_rgh"]
            u_star[1:-1, 1:-1] -= omega_p * self.dt * (p_corr[1:-1, 2:] - p_corr[1:-1, :-2]) * (0.5 * self.inv_dx)
            v_star[1:-1, 1:-1] -= omega_p * self.dt * (p_corr[2:, 1:-1] - p_corr[:-2, 1:-1]) * (0.5 * self.inv_dy)
            
            # Update pressure field with relaxation
            self.p += omega_p * p_corr
        
        # Final update of velocity fields
        self.u = u_star
        self.v = v_star

    def compute_nusselt_number(self) -> float:
        """
        Compute average Nusselt number at the hot wall.
        """
        # Compute temperature gradient at hot wall (left wall) in x-direction
        dT_dx = (self.T[:, 1] - self.T[:, 0]) / self.dx
        
        # Compute local Nusselt number
        Nu_local = -dT_dx  # Negative gradient due to heat flux direction
        
        # Compute average Nusselt number
        Nu_avg = np.mean(Nu_local)
        return Nu_avg

    def compute_stream_function(self) -> np.ndarray:
        """
        Compute stream function from velocity field.
        """
        # Compute vorticity: ω = dv/dx - du/dy
        vorticity = np.zeros_like(self.u)
        vorticity[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dx) -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * self.dy)
        )
        
        # Set up Poisson equation ∇²ψ = -vorticity with ψ=0 on boundaries
        n = self.ny * self.nx
        A = sparse.lil_matrix((n, n))
        b = np.zeros(n)
        def idx(i, j): return i * self.nx + j
        
        # Boundary conditions: ψ=0 on all walls
        for i in range(self.ny):
            for j in [0, self.nx-1]:  # Left and right walls
                A[idx(i, j), idx(i, j)] = 1.0
                b[idx(i, j)] = 0.0
        
        for j in range(self.nx):
            for i in [0, self.ny-1]:  # Bottom and top walls
                A[idx(i, j), idx(i, j)] = 1.0
                b[idx(i, j)] = 0.0
        
        # Interior points
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                current = idx(i, j)
                A[current, current] = -2 * (1 / self.dx**2 + 1 / self.dy**2)
                A[current, idx(i, j-1)] = 1 / self.dx**2
                A[current, idx(i, j+1)] = 1 / self.dx**2
                A[current, idx(i-1, j)] = 1 / self.dy**2
                A[current, idx(i+1, j)] = 1 / self.dy**2
                b[current] = -vorticity[i, j]
        
        # Solve the system
        A = A.tocsr()
        psi = splinalg.spsolve(A, b).reshape(self.ny, self.nx)
        return psi

    def _compute_time_step(self, max_velocity: float) -> float:
        return self.cfl * min(self.dx, self.dy) / max_velocity

    def solve(self):
        """
        Time-stepping solver with adaptive strategies and PISO algorithm.
        """
        time = 0.0
        iteration = 0
        
        # Storage for time series data
        time_series = {
            'time': [],
            'max_velocity': [],
            'nusselt_number': []
        }
        
        while time < self.total_time:
            # Adaptive time stepping
            max_velocity = np.max(np.sqrt(self.u**2 + self.v**2))
            
            if self.dt_fixed is None:
                self.dt = self._compute_time_step(max_velocity)
            else:
                self.dt = min(self.dt_fixed, self.total_time - time)
            
            # Solve governing equations
            self.solve_momentum_equations()
            self.solve_energy_equation()
            self.solve_piso_algorithm()  # Replace old pressure correction with PISO
            
            # Compute diagnostics
            nu_avg = self.compute_nusselt_number()
            
            # Store time series data
            time_series['time'].append(time)
            time_series['max_velocity'].append(max_velocity)
            time_series['nusselt_number'].append(nu_avg)
            
            # Update time
            time += self.dt
            iteration += 1
            
            # Optional: Print progress
            if iteration % 10 == 0:
                print(f"Time: {time:.4f}, Max Velocity: {max_velocity:.4e}, Nu: {nu_avg:.4f}")
        
        return time_series

    def plot_results(self, time_series=None):
        """
        Comprehensive visualization of simulation results.
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Velocity magnitude subplot
        plt.subplot(2, 3, 1)
        velocity_mag = np.sqrt(self.u**2 + self.v**2)
        plt.contourf(self.x, self.y, velocity_mag, cmap='viridis')
        plt.title('Velocity Magnitude')
        plt.colorbar()
        
        # Temperature field subplot
        plt.subplot(2, 3, 2)
        plt.contourf(self.x, self.y, self.T, cmap='inferno')
        plt.title('Temperature')
        plt.colorbar()
        
        # Stream function subplot
        plt.subplot(2, 3, 3)
        psi = self.compute_stream_function()
        plt.contourf(self.x, self.y, psi, cmap='coolwarm')
        plt.title('Stream Function')
        plt.colorbar()
        
        # Time series plots
        if time_series:
            plt.subplot(2, 3, 4)
            plt.plot(time_series['time'], time_series['max_velocity'])
            plt.title('Max Velocity vs Time')
            plt.xlabel('Time')
            plt.ylabel('Max Velocity')
            
            plt.subplot(2, 3, 5)
            plt.plot(time_series['time'], time_series['nusselt_number'])
            plt.title('Nusselt Number vs Time')
            plt.xlabel('Time')
            plt.ylabel('Nusselt Number')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main execution function.
    """
    # Create solver instance
    solver = NaturalConvectionSolverFV(
        nx=101,     # Grid points x
        ny=101,     # Grid points y
        Ra=1e6,     # Rayleigh number
        Pr=0.71,    # Prandtl number
        total_time=0.3,
        dt_fixed=0.00001,  # Force fixed time step 
        cfl=0.95,   # Total simulation time
        piso_iterations=2,  # Number of PISO iterations
        relaxation_factors={
            "U": 0.7,
            "T": 0.7,
            "p_rgh": 0.3
        }
    )
    
    # Solve simulation and get time series
    time_series = solver.solve()
    
    # Visualize results
    solver.plot_results(time_series)

if __name__ == "__main__":
    main()
