import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math
import traceback # For detailed error printing

# --- Default Parameters ---
DEFAULT_ALPHA = -1.0
DEFAULT_BETA = 1.0 # Changed default back to non-zero for testing shape
DEFAULT_K = 10.0
GRID_BOUND_FACTOR = 2.5
GRID_POINTS_2D = 350

# --- Helper Functions ---
def calculate_j2(s1, s2, s3):
    s1, s2, s3 = np.asarray(s1), np.asarray(s2), np.asarray(s3)
    return 0.5 * (s1**2 + s2**2 + s3**2)

def calculate_j3(s1, s2, s3):
    s1, s2, s3 = np.asarray(s1), np.asarray(s2), np.asarray(s3)
    return s1 * s2 * s3

# --- Yield Function Definition (Corrected except blocks) ---
def yield_func_kim_van(sigma1_in, sigma2_in, alpha, beta, k_pow_12):
    """
    Evaluates the Kim-Van yield function f=(J2)^6+alpha(J3)^4+beta(J2)^3 * (J3)^2 -k^12=0
    for the plane stress case (sigma3 = 0).
    Uses sigma1_in, sigma2_in to avoid confusion with deviatoric s1.
    """
    # Ensure inputs are numpy arrays
    sigma1_arr, sigma2_arr = np.asarray(sigma1_in), np.asarray(sigma2_in)
    sigma3_val = 0.0

    # Calculate hydrostatic stress
    sigma_h = (sigma1_arr + sigma2_arr + sigma3_val) / 3.0
    # Calculate deviatoric stresses
    s1 = sigma1_arr - sigma_h
    s2 = sigma2_arr - sigma_h
    s3 = sigma3_val - sigma_h # = -sigma_h for plane stress

    try:
        # Calculate invariants
        j2 = calculate_j2(s1, s2, s3)
        j3 = calculate_j3(s1, s2, s3)

        # Ensure J2 is non-negative before fractional power
        j2_safe = np.maximum(j2, 0.0)

        # Evaluate the function terms
        term1 = np.power(j2_safe, 6)
        term2 = alpha * np.power(j3, 4)
        # Only calculate term3 if beta is non-zero
        if abs(beta) > 1e-12: # Use tolerance for float comparison
             # Add small epsilon if J2 might be exactly zero and cause pow(0, 1.5) issues
             j2_safer_for_pow = np.maximum(j2_safe, 1e-15)
             term3 = beta * np.power(j2_safer_for_pow, 3) * np.power(j3, 2)
        else:
             term3 = 0.0 # If beta is zero, this term is zero

        f = term1 + term2 + term3 - k_pow_12

        # Replace inf/-inf with NaN as contour might handle NaN better
        f = np.where(np.isinf(f), np.nan, f)
        # Check for complex numbers (unlikely but possible)
        if np.any(np.iscomplex(f)):
            print("Warning: Complex numbers encountered. Taking real part.")
            f = np.nan_to_num(np.real(f), nan=1e10)

    except (FloatingPointError, ValueError, TypeError) as e:
        print(f"Warning: Numerical issue during yield function evaluation: {e}")
        # --- FIX: Use the array variable sigma1_arr for shape ---
        f = np.full_like(sigma1_arr, np.nan)
    except Exception as e:
        print(f"Unexpected error evaluating yield function: {e}")
        traceback.print_exc()
        # --- FIX: Use the array variable sigma1_arr for shape ---
        f = np.full_like(sigma1_arr, np.nan)
    return f

# --- Calculation Function for 2D Contour Data ---
def calculate_locus_data(alpha, beta, k, grid_bound_factor, n_points):
    """Generates grid and values for the 2D yield locus."""
    print(f"Calculating 2D: α={alpha}, β={beta}, k={k}, bound={grid_bound_factor*k}, n={n_points}")
    if k <= 0: raise ValueError("k > 0 required")
    if n_points < 3: raise ValueError("n_points >= 3 required")
    k_pow_12 = k**12; grid_bound = grid_bound_factor * k
    lin = np.linspace(-grid_bound, grid_bound, n_points)
    s1_grid_2d, s2_grid_2d = np.meshgrid(lin, lin, indexing='xy')
    # print(f"  2D Grid shape: {s1_grid_2d.shape}")
    try:
        f_values_2d = yield_func_kim_van(s1_grid_2d, s2_grid_2d, alpha, beta, k_pow_12)
        if np.all(np.isnan(f_values_2d)):
             raise ValueError("Yield function evaluation resulted in all NaNs.")
        min_f, max_f = np.nanmin(f_values_2d), np.nanmax(f_values_2d)
        if not (min_f < 1e-9 and max_f > -1e-9):
             print(f"Warning: Func range [{min_f:.2e}, {max_f:.2e}] may not cross zero.")
    except Exception as e:
        print(f"Error during 2D data calculation: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error calculating 2D data: {e}") # Pass original error message
    plot_lim_2d = grid_bound * 1.1
    return s1_grid_2d, s2_grid_2d, f_values_2d, plot_lim_2d

# --- Tkinter GUI Application ---
class YieldLocusApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive 2D Yield Locus (Kim-Van)")
        master.geometry("700x800")
        self.params = {}

        # --- Input Frame ---
        input_frame = ttk.Frame(master, padding="10"); input_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Parameters:", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        self.entries = {}; ttk.Label(input_frame, text="Alpha (α):").grid(row=1, column=0, sticky="w", padx=5)
        self.entries['alpha'] = ttk.Entry(input_frame, width=10); self.entries['alpha'].grid(row=1, column=1, padx=5); self.entries['alpha'].insert(0, str(DEFAULT_ALPHA))
        ttk.Label(input_frame, text="Beta (β):").grid(row=2, column=0, sticky="w", padx=5)
        self.entries['beta'] = ttk.Entry(input_frame, width=10); self.entries['beta'].grid(row=2, column=1, padx=5); self.entries['beta'].insert(0, str(DEFAULT_BETA))
        ttk.Label(input_frame, text="Yield Scale (k):").grid(row=3, column=0, sticky="w", padx=5)
        self.entries['k'] = ttk.Entry(input_frame, width=10); self.entries['k'].grid(row=3, column=1, padx=5); self.entries['k'].insert(0, str(DEFAULT_K))
        ttk.Label(input_frame, text="Grid Bound Factor:").grid(row=4, column=0, sticky="w", padx=5)
        self.entries['grid_bound_factor'] = ttk.Entry(input_frame, width=10); self.entries['grid_bound_factor'].grid(row=4, column=1, padx=5); self.entries['grid_bound_factor'].insert(0, str(GRID_BOUND_FACTOR))
        update_button = ttk.Button(input_frame, text="Update Plot", command=self.update_plot); update_button.grid(row=1, column=2, rowspan=4, padx=20)

        # --- Plot Frame ---
        plot_frame = ttk.Frame(master, padding="10"); plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.figure = plt.figure(figsize=(6, 6)); self.ax_2d = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame); self.canvas_widget = self.canvas.get_tk_widget(); self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame); self.toolbar.update(); self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_plot() # Initial plot

    def get_parameters(self):
        try:
            alpha = float(self.entries['alpha'].get()); beta = float(self.entries['beta'].get()); k = float(self.entries['k'].get())
            grid_bound_factor = float(self.entries['grid_bound_factor'].get())
            if k <= 0: raise ValueError("k > 0 required")
            if grid_bound_factor <= 0: raise ValueError("Grid Bound Factor > 0 required")
            params = {'alpha': alpha, 'beta': beta, 'k': k, 'grid_bound_2d': grid_bound_factor, 'n_points_2d': GRID_POINTS_2D}
            return params
        except ValueError as e: messagebox.showerror("Input Error", f"Invalid parameter: {e}"); return None

    def update_plot(self):
        params = self.get_parameters();
        if params is None: return
        self.ax_2d.clear()

        try:
            s1_2d, s2_2d, f_2d, plot_lim_2d = calculate_locus_data(
                params['alpha'], params['beta'], params['k'],
                params['grid_bound_2d'], params['n_points_2d']
            )
            # Validate data before plotting
            if s1_2d is None or np.all(np.isnan(f_2d)): raise ValueError("Calculation failed to produce valid data.")
            if s1_2d.shape != f_2d.shape or s2_2d.shape != f_2d.shape: raise ValueError(f"Shape mismatch.")
            if s1_2d.ndim != 2 or f_2d.ndim != 2: raise ValueError("Data must be 2D.")

            # Clean data for contouring
            f_2d_plot = np.nan_to_num(f_2d, nan=1e10, posinf=1e10, neginf=-1e10)

            print("  Attempting to plot 2D contour...")
            contour_set = self.ax_2d.contour(s1_2d, s2_2d, f_2d_plot, levels=[0.0], colors='green', linewidths=1.5)

            # Check contour success
            contours_found = hasattr(contour_set, 'allsegs') and len(contour_set.allsegs) > 0 and len(contour_set.allsegs[0]) > 0
            if not contours_found:
                 print("Warning: No contour lines found for f=0.")
                 self.ax_2d.text(0.5, 0.5, "No 2D locus found (f=0)\nTry adjusting Grid Bound Factor?", ha='center', va='center', color='orange', transform=self.ax_2d.transAxes)
                 plot_lim_2d = params['k'] * 1.5 # Use default limit

            # Formatting
            self.ax_2d.set_xlabel('$\sigma_1$ [MPa]'); self.ax_2d.set_ylabel('$\sigma_2$ [MPa]')
            self.ax_2d.set_title(f'Kim-Van 2D Locus ($\sigma_3=0$, α={params["alpha"]:.2f}, β={params["beta"]:.2f}, k={params["k"]:.1f})', fontsize=10)
            view_lim_2d = plot_lim_2d * 1.05
            self.ax_2d.set_xlim(-view_lim_2d, view_lim_2d); self.ax_2d.set_ylim(-view_lim_2d, view_lim_2d)
            self.ax_2d.axhline(0, color='grey', linestyle='--', linewidth=0.8); self.ax_2d.axvline(0, color='grey', linestyle='--', linewidth=0.8)
            self.ax_2d.grid(True, linestyle=':'); self.ax_2d.set_aspect('equal', adjustable='box')

        except Exception as e:
            print(f"ERROR during plot update: {e}")
            traceback.print_exc() # Print full traceback to console
            self.ax_2d.clear()
            self.ax_2d.set_title("Error Generating 2D Plot")
            # Display specific error on plot
            self.ax_2d.text(0.5, 0.5, f"Plot Error:\n{type(e).__name__}:\n{e}", ha='center', va='center', color='red', transform=self.ax_2d.transAxes, wrap=True, fontsize=8)

        self.figure.tight_layout(pad=1.5); self.canvas.draw()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk(); app = YieldLocusApp(root); root.mainloop()
