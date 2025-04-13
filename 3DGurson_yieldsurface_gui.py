import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math

# --- Default Parameters ---
DEFAULT_SIGMA_M = 1.0 # Matrix Yield Stress (scales the surface)
DEFAULT_CV = 0.01     # Void Volume Fraction
HYDROSTATIC_MIN_FACTOR = -2.5 # Factor of sigma_M (Gurson is sensitive here)
HYDROSTATIC_MAX_FACTOR = 1.5  # Factor of sigma_M
NUM_THETA = 50 # Points around surface
NUM_H = 60     # Points along hydrostatic axis (more needed for curved shape)

# --- Calculation Function ---
def calculate_gurson_points(sigma_m, cv, h_min_factor, h_max_factor, num_theta, num_h):
    """Calculates points on the Gurson yield surface."""
    if cv <= 0 or cv >= 1.0:
        raise ValueError("Void fraction Cv must be between 0 and 1")
    if sigma_m <= 0:
         raise ValueError("Matrix yield stress sigma_M must be positive")

    hydrostatic_min = h_min_factor * sigma_m
    hydrostatic_max = h_max_factor * sigma_m

    h = np.linspace(hydrostatic_min, hydrostatic_max, num_h)
    theta = np.linspace(0, 2 * np.pi, num_theta)
    h_grid, theta_grid = np.meshgrid(h, theta)

    # Calculate sigma_bar needed at each hydrostatic level h
    cosh_arg = (3.0 * h_grid) / (2.0 * sigma_m)
    # Avoid potential overflow for large arguments in cosh
    cosh_val = np.cosh(np.clip(cosh_arg, -700, 700)) # Clip argument to prevent overflow
    sigma_bar_sq_term = 1.0 + cv**2 - 2.0 * cv * cosh_val

    # Ensure term is non-negative before sqrt
    sigma_bar_sq_term = np.maximum(0, sigma_bar_sq_term)
    sigma_bar_yield = sigma_m * np.sqrt(sigma_bar_sq_term)

    # Calculate radius in deviatoric plane (r = sqrt(2/3)*sigma_bar)
    radius_h = np.sqrt(2.0 / 3.0) * sigma_bar_yield

    # Parametric equations converting (h, r(h), theta) to (sigma1, sigma2, sigma3)
    # Point = Hydrostatic Part + Deviatoric Part
    # Hydrostatic Part = (h, h, h)
    # Deviatoric Part = r(h) * [cos(theta)*u + sin(theta)*v]
    # u = (1/sqrt(2), -1/sqrt(2), 0)
    # v = (1/sqrt(6), 1/sqrt(6), -2/sqrt(6))
    sigma1 = h_grid + radius_h * (np.cos(theta_grid) / np.sqrt(2.0) + np.sin(theta_grid) / np.sqrt(6.0))
    sigma2 = h_grid + radius_h * (-np.cos(theta_grid) / np.sqrt(2.0) + np.sin(theta_grid) / np.sqrt(6.0))
    sigma3 = h_grid + radius_h * (-2.0 * np.sin(theta_grid) / np.sqrt(6.0))

    # Calculate limits for plotting axis line and box
    axis_lim_val = max(abs(hydrostatic_min), abs(hydrostatic_max))
    # Determine plot bounds based on calculated sigma values
    max_abs_sigma = np.max(np.abs([sigma1, sigma2, sigma3])) if sigma1.size > 0 else sigma_m * 1.5
    plot_lim = max(max_abs_sigma, axis_lim_val) * 1.1 # Ensure limits encompass data

    return sigma1, sigma2, sigma3, axis_lim_val, plot_lim

# --- Tkinter GUI Application ---
class GursonApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Gurson Yield Surface")
        master.geometry("750x800") # Adjust size

        self.params = {} # To store parameters

        # --- Input Frame ---
        input_frame = ttk.Frame(master, padding="10")
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Parameters:").grid(row=0, column=0, columnspan=3, sticky="w")

        # Input fields
        self.entries = {}
        ttk.Label(input_frame, text="Void Fraction (Cv):").grid(row=1, column=0, sticky="w", padx=5)
        self.entries['Cv'] = ttk.Entry(input_frame, width=10)
        self.entries['Cv'].grid(row=1, column=1, padx=5)
        self.entries['Cv'].insert(0, str(DEFAULT_CV))

        ttk.Label(input_frame, text="Matrix Yield (σM):").grid(row=2, column=0, sticky="w", padx=5)
        self.entries['sigma_m'] = ttk.Entry(input_frame, width=10)
        self.entries['sigma_m'].grid(row=2, column=1, padx=5)
        self.entries['sigma_m'].insert(0, str(DEFAULT_SIGMA_M))

        # Update Button
        update_button = ttk.Button(input_frame, text="Update Plot", command=self.update_plot)
        update_button.grid(row=1, column=2, rowspan=2, padx=20)


        # --- Plot Frame ---
        plot_frame = ttk.Frame(master, padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure = plt.figure(figsize=(7, 7))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Initial Plot ---
        self.update_plot() # Perform initial plot with defaults

    def get_parameters(self):
        """Reads and validates parameters from GUI."""
        try:
            cv = float(self.entries['Cv'].get())
            sigma_m = float(self.entries['sigma_m'].get())
            if not (0 < cv < 1):
                raise ValueError("Void fraction Cv must be between 0 and 1 (exclusive)")
            if sigma_m <= 0:
                 raise ValueError("Matrix yield stress sigma_M must be positive")
            # Use fixed calculation parameters for now
            params = {'Cv': cv, 'sigma_m': sigma_m,
                      'h_min_factor': HYDROSTATIC_MIN_FACTOR,
                      'h_max_factor': HYDROSTATIC_MAX_FACTOR,
                      'num_theta': NUM_THETA, 'num_h': NUM_H}
            return params
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter value: {e}")
            return None

    def update_plot(self):
        """Reads parameters, calculates points, and updates the plot."""
        params = self.get_parameters()
        if params is None:
            return # Stop if parameters invalid

        self.ax.clear() # Clear previous plot

        try:
            # Calculate points
            s1, s2, s3, axis_lim, plot_lim = calculate_gurson_points(
                params['sigma_m'], params['Cv'], params['h_min_factor'], params['h_max_factor'],
                params['num_theta'], params['num_h']
            )

            # Plot the yield surface
            self.ax.plot_surface(s1, s2, s3, cmap='plasma', alpha=0.75, edgecolor='k', linewidth=0.1, rstride=1, cstride=1)

            # Plot Hydrostatic Axis
            ax_h_coord = axis_lim / np.sqrt(3.0)
            self.ax.plot([0, ax_h_coord], [0, ax_h_coord], [0, ax_h_coord], 'g--', linewidth=1.5, label='Hydrostatic Axis')
            self.ax.plot([0, -ax_h_coord], [0, -ax_h_coord], [0, -ax_h_coord], 'g--', linewidth=1.5)

            # Axes Labels and Title
            self.ax.set_xlabel('$\sigma_1$')
            self.ax.set_ylabel('$\sigma_2$')
            self.ax.set_zlabel('$\sigma_3$')
            self.ax.set_title(f'Gurson Yield Surface (Cv={params["Cv"]:.3f}, $\sigma_M$={params["sigma_m"]:.1f})', fontsize=12)

            # Set Limits and Aspect Ratio
            view_lim = plot_lim * 1.1 # Ensure limits encompass the surface
            self.ax.set_xlim(-view_lim, view_lim)
            self.ax.set_ylim(-view_lim, view_lim)
            self.ax.set_zlim(-view_lim, view_lim)
            try:
                 self.ax.set_box_aspect([1,1,1]) # Equal scaling
            except AttributeError:
                 self.ax.axis('equal') # Fallback

            # Set initial view
            self.ax.view_init(elev=25., azim=-120)
            self.ax.legend()
            self.ax.grid(True, linestyle=':')

        except Exception as e:
            messagebox.showerror("Plotting Error", f"Could not plot surface: {e}")
            # Clear axes if plotting failed
            self.ax.clear()
            self.ax.set_title("Error Generating Plot")
            self.ax.text(0.5, 0.5, 0.5, "Error", ha='center', va='center', transform=self.ax.transAxes)


        # Redraw the canvas
        self.canvas.draw()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GursonApp(root)
    root.mainloop()
