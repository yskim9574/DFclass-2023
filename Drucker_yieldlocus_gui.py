
# --- Default Parameters ---
DEFAULT_ALPHA = 0.1   # Drucker parameter
DEFAULT_K = 1.0       # Yield scale parameter
GRID_BOUND_2D = 2.0   # Grid extent for 2D plot (relative to k)
GRID_POINTS_2D = 150  # Points per dim for 2D contour

# --- Helper Functions (Assume these are correct as they didn't cause errors before) ---
def calculate_j2(s1, s2, s3):
    s1, s2, s3 = np.asarray(s1), np.asarray(s2), np.asarray(s3)
    return 0.5 * (s1**2 + s2**2 + s3**2)

def calculate_j3(s1, s2, s3):
    s1, s2, s3 = np.asarray(s1), np.asarray(s2), np.asarray(s3)
    return s1 * s2 * s3

def drucker_yield_function(sigma1, sigma2, sigma3, alpha, k_pow_6):
    sigma1, sigma2, sigma3 = np.asarray(sigma1), np.asarray(sigma2), np.asarray(sigma3)
    sigma_h = (sigma1 + sigma2 + sigma3) / 3.0
    s1 = sigma1 - sigma_h; s2 = sigma2 - sigma_h; s3 = sigma3 - sigma_h
    j2 = calculate_j2(s1, s2, s3)
    j3 = calculate_j3(s1, s2, s3)
    j2_clipped = np.clip(j2, -1e10, 1e10) # Prevent overflow potential
    try:
        f = j2_clipped**3 - alpha * (j3**2) - k_pow_6
        # Check for complex numbers resulting from operations if necessary (unlikely here)
        if np.any(np.iscomplex(f)):
            print("Warning: Complex numbers encountered during Drucker evaluation.")
            f = np.nan_to_num(np.real(f), nan=1e10) # Take real part, replace NaN
    except FloatingPointError:
        print("Warning: Floating point error during Drucker evaluation.")
        f = np.full_like(j2, np.nan) # Use np.full_like to get correct shape
    return f

# --- Calculation Function for 2D Contour Data ---
def calculate_drucker_locus_2d_data(alpha, k, grid_bound_factor, n_points):
    """Generates grid and values for the 2D yield locus (sigma3=0)."""
    print(f"Calculating 2D locus data: alpha={alpha}, k={k}")
    if k <= 0: raise ValueError("k must be positive")
    if n_points < 3: raise ValueError("n_points for 2D must be at least 3 for contour") # Contour needs >2x2
    k_pow_6 = k**6
    grid_bound = grid_bound_factor * k
    lin = np.linspace(-grid_bound, grid_bound, n_points)
    s1_grid_2d, s2_grid_2d = np.meshgrid(lin, lin, indexing='xy')
    s3_val = 0.0
    print(f"  2D Grid shape: {s1_grid_2d.shape}")
    try:
        f_values_2d = drucker_yield_function(s1_grid_2d, s2_grid_2d, s3_val, alpha, k_pow_6)
        # Check for potential issues before returning
        if np.all(np.isnan(f_values_2d)):
            print("Error: All calculated 2D Drucker values are NaN.")
            raise ValueError("All 2D Drucker values are NaN")

        # Check if function crosses zero (more robustly)
        min_f, max_f = np.nanmin(f_values_2d), np.nanmax(f_values_2d)
        if min_f > 0 or max_f < 0:
             print(f"Warning: Drucker function might not cross zero on 2D grid (Range: [{min_f:.2e}, {max_f:.2e}]).")
             # It's okay to return, contour will handle it, but good to warn.

    except Exception as e:
        print(f"Error evaluating 2D Drucker function: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error evaluating 2D Drucker function: {e}")

    plot_lim_2d = grid_bound * 1.1
    return s1_grid_2d, s2_grid_2d, f_values_2d, plot_lim_2d

# --- Tkinter GUI Application ---
class Drucker2DApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Drucker 2D Yield Locus")
        master.geometry("700x800")
        self.params = {}

        # --- Input Frame ---
        input_frame = ttk.Frame(master, padding="10"); input_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Parameters:", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        self.entries = {}
        ttk.Label(input_frame, text="Alpha (α):").grid(row=1, column=0, sticky="w", padx=5)
        self.entries['alpha'] = ttk.Entry(input_frame, width=10); self.entries['alpha'].grid(row=1, column=1, padx=5)
        self.entries['alpha'].insert(0, str(DEFAULT_ALPHA))
        ttk.Label(input_frame, text="Yield Scale (k):").grid(row=2, column=0, sticky="w", padx=5)
        self.entries['k'] = ttk.Entry(input_frame, width=10); self.entries['k'].grid(row=2, column=1, padx=5)
        self.entries['k'].insert(0, str(DEFAULT_K))
        update_button = ttk.Button(input_frame, text="Update Plot", command=self.update_plot)
        update_button.grid(row=1, column=2, rowspan=2, padx=20)

        # --- Plot Frame ---
        plot_frame = ttk.Frame(master, padding="10"); plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.figure = plt.figure(figsize=(6, 6))
        self.ax_2d = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget(); self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame); self.toolbar.update(); self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_plot() # Initial plot

    def get_parameters(self):
        try:
            alpha = float(self.entries['alpha'].get())
            k = float(self.entries['k'].get())
            if k <= 0: raise ValueError("Yield scale k must be positive")
            params = {'alpha': alpha, 'k': k,
                      'grid_bound_2d': GRID_BOUND_2D,
                      'n_points_2d': GRID_POINTS_2D}
            return params
        except ValueError as e: messagebox.showerror("Input Error", f"Invalid parameter: {e}"); return None

    def update_plot(self):
        params = self.get_parameters();
        if params is None: return
        self.ax_2d.clear() # Clear previous plot

        try:
            # --- Calculate 2D Plot Data ---
            s1_2d, s2_2d, f_2d, plot_lim_2d = calculate_drucker_locus_2d_data(
                params['alpha'], params['k'], params['grid_bound_2d'], params['n_points_2d']
            )

            # --- Input Data Validation before Plotting ---
            if s1_2d is None or s2_2d is None or f_2d is None:
                 raise ValueError("Calculation function returned None for grid data.")
            if np.any(np.isnan(s1_2d)) or np.any(np.isnan(s2_2d)):
                 raise ValueError("Grid contains NaN values.")
             # Allow NaNs in f_2d, contour should handle them generally
            # if np.any(np.isnan(f_2d)):
            #      print("Warning: f_2d contains NaNs, contour results may be affected.")
            if s1_2d.shape != f_2d.shape or s2_2d.shape != f_2d.shape:
                 raise ValueError(f"Shape mismatch for contour: S1{s1_2d.shape}, S2{s2_2d.shape}, F{f_2d.shape}")
            if s1_2d.ndim != 2 or f_2d.ndim != 2:
                 raise ValueError("Data passed to contour must be 2-dimensional.")


            # --- Plotting 2D Contour ---
            print("  Attempting to plot 2D contour...") # Debug print
            contour_set = self.ax_2d.contour(s1_2d, s2_2d, f_2d, levels=[0.0], colors='red', linewidths=2)

            # --- Check if Contour Lines were Found ---
            # The reliable check is often len(contour_set.allsegs[0]) > 0 if levels=[0.0]
            # where allsegs is a list (one element per level) of lists of segments (paths).
            contours_found = False
            if hasattr(contour_set, 'allsegs') and len(contour_set.allsegs) > 0 and len(contour_set.allsegs[0]) > 0:
                contours_found = True
                print("  Contour lines found.") # Debug print
            # else:
                # Alternative check if allsegs is structured differently
                # if hasattr(contour_set, 'collections') and contour_set.collections:
                #     contours_found = True

            if not contours_found:
                 print("Warning: No contour lines found for f=0 by ax.contour.")
                 self.ax_2d.text(0.5, 0.5, "No 2D locus found (f=0)", ha='center', va='center', color='orange', transform=self.ax_2d.transAxes) # Use relative coords
                 # Use default limit if no contour found
                 plot_lim_2d = params['k'] * 1.5

            # --- 2D Plot Formatting ---
            self.ax_2d.set_xlabel('$\sigma_1$'); self.ax_2d.set_ylabel('$\sigma_2$')
            self.ax_2d.set_title(f'Drucker 2D Locus ($\sigma_3=0$, α={params["alpha"]:.2f}, k={params["k"]:.1f})', fontsize=11)
            view_lim_2d = plot_lim_2d * 1.1
            self.ax_2d.set_xlim(-view_lim_2d, view_lim_2d); self.ax_2d.set_ylim(-view_lim_2d, view_lim_2d)
            self.ax_2d.axhline(0, color='grey', linestyle='--', linewidth=0.8); self.ax_2d.axvline(0, color='grey', linestyle='--', linewidth=0.8)
            self.ax_2d.grid(True, linestyle=':'); self.ax_2d.set_aspect('equal', adjustable='box')

        except Exception as e:
            print(f"ERROR during 2D plot generation: {e}") # Print error to console
            traceback.print_exc() # Print full traceback
            self.ax_2d.clear() # Clear axes on error
            self.ax_2d.set_title("Error Generating 2D Plot")
            # Display the actual error message 'e' on the plot using relative coordinates
            self.ax_2d.text(0.5, 0.05, f"2D Plot Error:\n{type(e).__name__}: {e}", ha='center', va='bottom', color='red', transform=self.ax_2d.transAxes, wrap=True, fontsize=8)


        # Adjust layout and redraw
        self.figure.tight_layout(pad=1.5) # Adjusted padding
        self.canvas.draw()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = Drucker2DApp(root)
    root.mainloop()
