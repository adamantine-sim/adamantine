import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import pyvista as pv
import os
from cycler import cycler

# --- Physical and Problem Parameters ---
# (Adjust these values to test different scenarios)
L = 0.1      # Slab thickness (m)
k = 1.0     # Thermal conductivity (W/m.K)
h = 50     # Convective heat transfer coefficient (W/m^2.K)
T_inf = 300   # Ambient fluid temperature (°C)
rho = 100   # Density (kg/m^3) - Needed for transient
cp = 100     # Specific heat capacity (J/kg.K) - Needed for transient
T_i = 400  # Initial temperature (°C)[1, 10, 50, 100, 200, 1000] - Needed for transient

# Derived parameters
alpha = k / (rho * cp) # Thermal diffusivity (m^2/s)
Bi = h * L / k         # Biot number

# --- 1D Transient Cooling of a Slab ---

def _eigenvalue_eqn(zeta, Bi):
    """Equation whose roots are the eigenvalues: zeta*tan(zeta) - Bi = 0"""
    # Handle zeta close to multiples of pi/2 where tan explodes
    if np.isclose(np.cos(zeta), 0):
        return np.inf * np.sign(zeta * np.tan(zeta)) # Return large number with correct sign
    return zeta * np.tan(zeta) - Bi

def find_eigenvalues(Bi, n_terms):
    """Finds the first n_terms positive roots (eigenvalues) of zeta*tan(zeta) = Bi."""
    eigenvalues = np.zeros(n_terms)
    if Bi < 0:
        raise ValueError("Biot number Bi must be non-negative for this solution.")
    if n_terms <= 0:
        return np.array([])

    # Find roots numerically using root_scalar
    for n in range(1, n_terms + 1):
        # Define search interval for the n-th root
        # The n-th root lies between (n-1)*pi and (n-1)*pi + pi/2 = (n-0.5)*pi
        # Add small epsilon to avoid boundaries where tan might be undefined/problematic
        a = (n - 1) * np.pi + 1e-9
        b = (n - 0.5) * np.pi - 1e-9

        # Handle the Bi=0 case separately (roots are 0, pi, 2pi, ... but we need positive roots for the formula)
        # The standard formula assumes Bi > 0. If Bi is very close to 0, root finding might be unstable.
        # However, zeta*tan(zeta)=0 has roots at n*pi. But the C_n formula needs care.
        # Let's assume Bi > small_value for the numerical search.
        # For the first root (n=1), the interval is (0, pi/2)
        if n == 1 and a < 1e-9: a = 1e-9 # Avoid exactly zero for tan

        try:
            sol = root_scalar(_eigenvalue_eqn, args=(Bi,), bracket=[a, b], method='brentq')
            if sol.converged:
                eigenvalues[n-1] = sol.root
            else:
                print(f"Warning: Root finding did not converge for n={n}. Bi={Bi}")
                # Attempt a wider search for robustness, though less precise interval
                a_wide = (n - 1) * np.pi + 1e-9
                b_wide = n * np.pi - 1e-9
                try:
                    sol = root_scalar(_eigenvalue_eqn, args=(Bi,), bracket=[a_wide, b_wide], method='brentq')
                    if sol.converged:
                         eigenvalues[n-1] = sol.root
                    else:
                         eigenvalues[n-1] = np.nan # Mark as failed
                except ValueError: # Bracket might not contain a root if Bi is large
                     eigenvalues[n-1] = np.nan
                     print(f"Error finding root in wider bracket for n={n}. Bi={Bi}")

        except ValueError as e:
            # This might happen if the function has the same sign at both ends of the bracket
            # or if the bracket is invalid.
            print(f"Error finding root for n={n}, Bi={Bi}: {e}. Bracket=[{a}, {b}]")
            eigenvalues[n-1] = np.nan # Mark as failed

    # Filter out any NaNs if convergence failed
    eigenvalues = eigenvalues[~np.isnan(eigenvalues)]
    if len(eigenvalues) < n_terms:
        print(f"Warning: Only found {len(eigenvalues)} valid eigenvalues out of {n_terms} requested.")
    return eigenvalues

def calculate_coefficients(zeta_n):
    """Calculates the C_n coefficients for the transient solution."""
    # Can rewrite denominator using double angle: zeta_n + 0.5*sin(2*zeta_n)
    sin_zeta = np.sin(zeta_n)
    cos_zeta = np.cos(zeta_n)
    C_n = (2.0 * sin_zeta) / (zeta_n + sin_zeta * cos_zeta)
    return C_n

def solve_transient_1d(x, t, L, alpha, Bi, T_inf, T_i, n_terms=50):
    """
    Calculates the transient temperature distribution in a 1D slab
    with convection at x=0, adiabatic at x=L.

    Args:
        x (np.ndarray or float): Position(s) along the slab (m).
        t (float): Time (s).
        L (float): Slab thickness (m).
        alpha (float): Thermal diffusivity (m^2/s).
        Bi (float): Biot number (hL/k).
        T_inf (float): Ambient fluid temperature (°C).
        T_i (float): Initial temperature (°C).
        n_terms (int): Number of terms to sum in the infinite series.

    Returns:
        np.ndarray or float: Temperature(s) at position(s) x and time t (°C).
    """
    if alpha <= 0:
        raise ValueError("Thermal diffusivity alpha must be positive.")
    if t < 0:
        raise ValueError("Time t cannot be negative.")
    if Bi < 0:
         raise ValueError("Biot number Bi cannot be negative.")

    # Handle trivial case t=0
    if t == 0:
        return np.ones_like(x) * T_i if isinstance(x, np.ndarray) else T_i

    # Handle Bi = 0 (fully insulated) -> T(x,t) = T_i (no heat transfer)
    if Bi == 0:
         return np.ones_like(x) * T_i if isinstance(x, np.ndarray) else T_i

    theta_i = T_i - T_inf
    if np.isclose(theta_i, 0): # Initial temp is same as ambient
        return np.ones_like(x) * T_inf if isinstance(x, np.ndarray) else T_inf

    # Find eigenvalues and coefficients
    zeta_n = find_eigenvalues(Bi, n_terms)
    if len(zeta_n) == 0:
        print("Error: No eigenvalues found. Cannot compute transient solution.")
        return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan
    C_n = calculate_coefficients(zeta_n)

    # Calculate dimensionless time (Fourier number)
    Fo = alpha * t / L**2

    # Sum the series
    #theta_ratio = np.zeros_like(x, dtype=float) # Dimensionless temp ratio theta/theta_i
    #x_norm = (L - x) / L # Normalized coordinate for the cosine term
    x_norm = x / L # Normalized coordinate for the cosine term

    # Use broadcasting for efficiency
    zeta_n_col = zeta_n[:, np.newaxis] # Shape (n_terms, 1)
    C_n_col = C_n[:, np.newaxis]       # Shape (n_terms, 1)
    exp_term = np.exp(-zeta_n_col**2 * Fo) # Shape (n_terms, 1)
    cos_term = np.cos(zeta_n_col * x_norm) # Shape (n_terms, n_x)

    # Sum over the terms (axis=0)
    # theta_ratio = Sum[ C_n * exp * cos ]
    series_sum = np.sum(C_n_col * exp_term * cos_term, axis=0) # Shape (n_x,)

    # Calculate final temperature T = theta + T_inf = theta_ratio * theta_i + T_inf
    T = series_sum * theta_i + T_inf
    #print(max(T))
    return T

def solve_transient_1d_full_wall(x, t, L, alpha, Bi, T_inf, T_i, n_terms=50):
    """
    Analytical solution for a full plane wall of thickness 2L with convection on both surfaces.
    x ranges from 0 to L due to symmetry.
    """
    if alpha <= 0:
        raise ValueError("Thermal diffusivity alpha must be positive.")
    if t < 0:
        raise ValueError("Time t cannot be negative.")
    if Bi < 0:
        raise ValueError("Biot number Bi cannot be negative.")

    if t == 0:
        return np.ones_like(x) * T_i
    if Bi == 0:
        return np.ones_like(x) * T_i

    theta_i = T_i - T_inf
    if np.isclose(theta_i, 0):
        return np.ones_like(x) * T_inf
    print(n_terms)
    zeta_n = find_eigenvalues(Bi, n_terms)
    if len(zeta_n) == 0:
        print("Error: No eigenvalues found.")
        return np.nan * np.ones_like(x)

    # New coefficients for full plane wall (Eq. 5.24 in the book)
    C_n = 4 * np.sin(zeta_n) / (2 * zeta_n + np.sin(2 * zeta_n))

    Fo = alpha * t / L**2
    x_norm = x / L  # normalized position x* = x/L

    zeta_n_col = zeta_n[:, np.newaxis]
    C_n_col = C_n[:, np.newaxis]
    exp_term = np.exp(-zeta_n_col**2 * Fo)
    cos_term = np.cos(zeta_n_col * x_norm)

    series_sum = np.sum(C_n_col * exp_term * cos_term, axis=0)
    T = series_sum * theta_i + T_inf
    return T

# --- Simulation Reader ---
def get_simulation_temperature_at_time(t_seconds, dt_simulation, L, resolution, pvtu_folder, solname, field_name="temperature"):
    step = int(round(t_seconds / dt_simulation))
    file_path = os.path.join(pvtu_folder, f"{solname}.{step}.pvtu")
    if not os.path.exists(file_path):
        print(f"[WARNING] Missing file: {file_path}")
        return None, None

    mesh = pv.read(file_path)
    if field_name not in mesh.array_names:
        print(f"[WARNING] Field '{field_name}' not found.")
        return None, None

    sampled = mesh.sample_over_line(pointa=(0, 0.05, 0.05), pointb=(L, 0.05, 0.05), resolution=resolution)
    coords = sampled.points
    temps = sampled[field_name]

    mask = temps > 1e-6
    #x_sim = coords[mask, 0] - L  # center about 0
    x_sim = coords[mask, 0]  # center about 0
    temps_sim = temps[mask]

    return x_sim, temps_sim

# --- Plotting Functions ---
def plot_combined_analytical_and_sim(times_seconds, dt_sim, L, k, h, T_inf, alpha, Bi, T_i, n_terms=50, pvtu_folder=".", solname = "solution", field_name="temperature"):
    #x_vals = np.linspace(-L, L, 201)
    x_vals = np.linspace(0, L, 201)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    time_to_color = dict(zip(times_seconds, color_cycle))

    for t, color in zip(times_seconds, color_cycle):
        #T_analytic = solve_transient_1d_full_wall(x_vals, t, L, alpha, Bi, T_inf, T_i, n_terms)
        T_analytic = solve_transient_1d(x_vals, t, L, alpha, Bi, T_inf, T_i, n_terms)
        plt.plot(x_vals, T_analytic, label=f"t = {t} s", color=color, linestyle='-')

        x_sim, T_sim = get_simulation_temperature_at_time(t, dt_sim, L, 201, pvtu_folder, solname, field_name)
        if x_sim is not None:
            plt.plot(x_sim, T_sim, 'x-', label=f"t = {t} s (Sim)", color=color)

    plt.xlabel("x (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Analytical vs Simulation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_error_vs_x(times_seconds, dt_sim, L, alpha, Bi, T_inf, T_i, n_terms=50, pvtu_folder=".", solname = "solution", field_name="temperature"):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(8, 5))

    for t, color in zip(times_seconds, color_cycle):
        x_sim, T_sim = get_simulation_temperature_at_time(t, dt_sim, L, 500, pvtu_folder, solname, field_name)
        if x_sim is None:
            continue

        #T_analytic = solve_transient_1d_full_wall(x_sim, t, L, alpha, Bi, T_inf, T_i, n_terms)
        T_analytic = solve_transient_1d(x_sim, t, L, alpha, Bi, T_inf, T_i, n_terms)
        error = np.abs(T_sim - T_analytic)

        plt.plot(x_sim, error, label=f"t = {t} s", color=color, marker='o', markersize=2, linewidth=0.7)

    plt.xlabel("x (m)")
    plt.ylabel("Absolute Error (K)")
    plt.title("Error vs Position")
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_max_error_vs_time(L, alpha, Bi, T_inf, T_i, n_terms, 
                           pvtu_folder, filename, timestep_scale, times, label, 
                           field_name="temperature"):
    """
    Plot max(abs error) vs time for a given simulation set (coarse/fine).
    """
    max_errors = []
    time_values = []

    for t_val in times:
        step = int(t_val / timestep_scale)
        file_path = os.path.join(pvtu_folder, f"{filename}.{step}.pvtu")
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        mesh = pv.read(file_path)
        if field_name not in mesh.array_names:
            print(f"Field '{field_name}' not in mesh.")
            continue

        sampled = mesh.sample_over_line(pointa=(0, 0.05, 0.05), pointb=(L, 0.05, 0.05), resolution=1000)
        coords = sampled.points
        temps_sim = sampled[field_name]

        mask = temps_sim > 1e-6
        #x_sim = coords[mask, 0] - L
        x_sim = coords[mask, 0]
        temps_sim = temps_sim[mask]

        #T_analytic = solve_transient_1d_full_wall(x_sim, t_val, L, alpha, Bi, T_inf, T_i, n_terms)
        T_analytic = solve_transient_1d(x_sim, t_val, L, alpha, Bi, T_inf, T_i, n_terms)
        error = np.abs(temps_sim - T_analytic)

        max_errors.append(np.max(error))
        time_values.append(t_val)

    plt.plot(time_values, max_errors, marker='o', label=label)

def plot_simulation_difference(times_seconds,
                               dt1, pvtu_folder1, solname1,
                               dt2, pvtu_folder2, solname2,
                               L, resolution=500,
                               field_name="temperature"):
    """
    Plot the pointwise temperature difference between two simulation runs
    (e.g. fine_sim_nBC minus fine_sim) along the x–axis at the specified times.
    """
   
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(8, 5))

    for t, color in zip(times_seconds, colors):
        # load sim1
        x1, T1 = get_simulation_temperature_at_time(t, dt1, L, resolution,
                                                    pvtu_folder1, solname1, field_name)
        # load sim2
        x2, T2 = get_simulation_temperature_at_time(t, dt2, L, resolution,
                                                    pvtu_folder2, solname2, field_name)
        if x1 is None or x2 is None:
            continue

        # interpolate both onto a common grid
        x_common = np.linspace(-L, L, resolution)
        T1_i = np.interp(x_common, x1, T1)
        T2_i = np.interp(x_common, x2, T2)

        dT = T1_i - T2_i
        dx = x_common[1] - x_common[0]
        L2_norm = np.sqrt(np.sum(dT**2) * dx)
        print(f"t = {t:>5}s → L2 norm of dT = {L2_norm:.3e}")
        plt.plot(x_common, dT,
                 label=f"dT at t={t}s",
                 color=color,
                 linewidth=1.2)

    plt.axhline(0, color='k', lw=0.5, linestyle='--')
    plt.xlabel("x (m)")
    plt.ylabel("T[fine_nBC] – T[fine] (K)")
    plt.title("Difference between fine_sim_nBC and fine_sim")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    times_seconds = [1, 10, 50, 100, 200, 1000]
    dt_simulation = 2.5e-4
    pvtu_folder1 = "." #"halfslab_tinyt" #"fine_sim_nBC/"
    pvtu_folder2 = "halfslab_nBC_lessfine_reddt"
    pvtu_folder3 = "fine_sim_nBC_convad"
    filename1 = "solution"
    filename2 = "solution"
    filename3 = "solution_dblres"
    field_name = "temperature"
    n_terms = 800
   
    plot_simulation_difference(
        times_seconds,
        dt1=7.8125e-5, pvtu_folder1=pvtu_folder2, solname1 = filename2,
        dt2=1e-5, pvtu_folder2=pvtu_folder1, solname2 = filename2,
        L=L, resolution=500,
        field_name="temperature"
    )
   
    
    plot_combined_analytical_and_sim(
        times_seconds, dt_simulation,
        L, k, h, T_inf, alpha, Bi, T_i,
        n_terms=n_terms,
        pvtu_folder=pvtu_folder1,
        field_name=field_name
    )

    plot_error_vs_x(
        times_seconds, dt_simulation,
        L, alpha, Bi, T_inf, T_i,
        n_terms=n_terms,
        pvtu_folder=pvtu_folder1,
        field_name=field_name
    )
    plt.figure(figsize=(7, 5))
    
    plot_max_error_vs_time(L=0.1, alpha=alpha, Bi=Bi, T_inf=T_inf, T_i=T_i, n_terms=201,
                           pvtu_folder=pvtu_folder2, filename=filename2, timestep_scale=1e-3, times=times_seconds,
                           label="Coarse (dt=1e-3, dx=2.5e-3)")
    
    plot_max_error_vs_time(L=0.1, alpha=alpha, Bi=Bi, T_inf=T_inf, T_i=T_i, n_terms=201,
                           pvtu_folder=pvtu_folder1, filename=filename1, timestep_scale=dt_simulation, times=times_seconds,
                           label="Fine (dt=1e-4, dx=1.25e-3)")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Time (s)")
    plt.ylabel("Max Absolute Error (K)")
    plt.title("Max Error vs Time (log-log)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
