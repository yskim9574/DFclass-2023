import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font, filedialog # Added filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import least_squares
import warnings
import os # Added os for path operations

# --- Core Optimization Logic (Unchanged from previous corrected version) ---

def j2(s1, s2):
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    return (1/3) * (s1**2 - s1*s2 + s2**2)

def j3(s1, s2):
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    return (1/27) * (2*s1**3 + 2*s2**3 - 3*(s1+s2)*s1*s2)

def yield_func(s1, s2, alpha, beta, k):
    J2 = j2(s1, s2)
    J3 = j3(s1, s2)
    J2_safe = np.maximum(J2, 0)
    term1 = J2_safe**3
    term2 = alpha * (J3**2)
    term3 = beta * J2_safe * np.sqrt(J2_safe) * J3
    if k <= 0:
        return np.full_like(J2, 1e18)
    return term1 + term2 + term3 - k**6

def residuals(params, s1, s2):
    alpha, beta, k = params
    if k <= 1e-9:
        return np.full_like(s1, 1e18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = yield_func(s1, s2, alpha, beta, k)
    return np.nan_to_num(result, nan=1e18, posinf=1e18, neginf=-1e18)

# --- GUI Application Class (Modified for File Loading) ---

class YieldLocusOptimizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Kim-Van Yield Locus Optimizer (File Load)")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=10)
        master.option_add("*Font", default_font)

        # --- Data Storage ---
        # StringVars for displaying data in Entry widgets
        self.sigma1_exp_str = tk.StringVar(value="10, 13, 14, 10.1, 7.0, 0, -13, -11.3, 0")
        self.sigma2_exp_str = tk.StringVar(value="0, 7, 14, 14.8, 13.75, 10, 0, -11.3, -13.05")
        # Store actual numpy arrays internally after loading/parsing
        self.sigma1_exp_data = self.parse_data(self.sigma1_exp_str.get())
        self.sigma2_exp_data = self.parse_data(self.sigma2_exp_str.get())

        self.data_filepath_str = tk.StringVar(value="") # To store the path of the loaded file

        self.alpha_init_str = tk.StringVar(value="-1.0")
        self.beta_init_str = tk.StringVar(value="1.0")
        self.k_init_str = tk.StringVar(value="10.0")
        self.k_lower_bound_str = tk.StringVar(value="1e-6")

        self.alpha_opt_str = tk.StringVar(value="N/A")
        self.beta_opt_str = tk.StringVar(value="N/A")
        self.k_opt_str = tk.StringVar(value="N/A")
        self.cost_str = tk.StringVar(value="N/A")
        self.status_str = tk.StringVar(value="Ready. Load data or use defaults.")

        # --- Layout Frames ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_panel.rowconfigure(9, weight=1) # Adjusted row index for status bar weight

        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)

        # --- Left Panel Widgets ---
        row_idx = 0

        # Data Loading Section (NEW)
        load_frame = ttk.LabelFrame(left_panel, text="Load Experimental Data File", padding="5")
        load_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10)); row_idx += 1
        load_frame.columnconfigure(1, weight=1)

        ttk.Label(load_frame, text="File:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.file_entry = ttk.Entry(load_frame, textvariable=self.data_filepath_str, state='readonly', width=30)
        self.file_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.browse_button = ttk.Button(load_frame, text="Browse...", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, sticky="e", padx=(0,5), pady=2)
        self.load_button = ttk.Button(load_frame, text="Load Data", command=self.load_data_action, width=10)
        self.load_button.grid(row=1, column=1, columnspan=2, sticky="e", padx=5, pady=5)


        # Experimental Data Display Section (Modified)
        data_frame = ttk.LabelFrame(left_panel, text="Current Data (Editable after Load)", padding="5")
        data_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10)); row_idx += 1
        data_frame.columnconfigure(1, weight=1)
        ttk.Label(data_frame, text="σ₁:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.sigma1_entry = ttk.Entry(data_frame, textvariable=self.sigma1_exp_str, width=40)
        self.sigma1_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(data_frame, text="σ₂:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.sigma2_entry = ttk.Entry(data_frame, textvariable=self.sigma2_exp_str, width=40)
        self.sigma2_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        # Add a note about editing
        edit_note = ttk.Label(data_frame, text="Edit here to override loaded file data.", style='secondary.TLabel')
        edit_note.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=(0,2))
        self.style.configure('secondary.TLabel', foreground='gray')


        # Initial Guess Section
        guess_frame = ttk.LabelFrame(left_panel, text="Initial Guess & Bounds", padding="5")
        guess_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10)); row_idx += 1
        guess_frame.columnconfigure(1, weight=1)
        # (Widgets inside guess_frame remain the same as before)
        ttk.Label(guess_frame, text="α₀:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.alpha_init_entry = ttk.Entry(guess_frame, textvariable=self.alpha_init_str, width=15)
        self.alpha_init_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(guess_frame, text="β₀:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.beta_init_entry = ttk.Entry(guess_frame, textvariable=self.beta_init_str, width=15)
        self.beta_init_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(guess_frame, text="k₀:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.k_init_entry = ttk.Entry(guess_frame, textvariable=self.k_init_str, width=15)
        self.k_init_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(guess_frame, text="k Lower Bound:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.k_bound_entry = ttk.Entry(guess_frame, textvariable=self.k_lower_bound_str, width=15)
        self.k_bound_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(guess_frame, text="(α, β bounds are +/- ∞)").grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Control Button
        self.optimize_button = ttk.Button(left_panel, text="Optimize Parameters", command=self.run_optimization)
        self.optimize_button.grid(row=row_idx, column=0, sticky="ew", pady=10); row_idx += 1

        # Results Section
        results_frame = ttk.LabelFrame(left_panel, text="Optimized Parameters", padding="5")
        results_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 5)); row_idx += 1
        results_frame.columnconfigure(1, weight=1)
        # (Widgets inside results_frame remain the same as before)
        ttk.Label(results_frame, text="α_opt:").grid(row=0, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(results_frame, textvariable=self.alpha_opt_str).grid(row=0, column=1, sticky="ew", padx=5, pady=1)
        ttk.Label(results_frame, text="β_opt:").grid(row=1, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(results_frame, textvariable=self.beta_opt_str).grid(row=1, column=1, sticky="ew", padx=5, pady=1)
        ttk.Label(results_frame, text="k_opt:").grid(row=2, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(results_frame, textvariable=self.k_opt_str).grid(row=2, column=1, sticky="ew", padx=5, pady=1)


        # Cost Section
        cost_frame = ttk.LabelFrame(left_panel, text="Goodness of Fit", padding="5")
        cost_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10)); row_idx += 1
        cost_frame.columnconfigure(1, weight=1)
        # (Widgets inside cost_frame remain the same as before)
        ttk.Label(cost_frame, text="Cost (½Σres²):").grid(row=0, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(cost_frame, textvariable=self.cost_str).grid(row=0, column=1, sticky="ew", padx=5, pady=1)

        # Status Bar
        status_frame = ttk.Frame(left_panel)
        status_frame.grid(row=row_idx, column=0, sticky="sew", pady=(10, 0))
        left_panel.rowconfigure(row_idx, weight=1) # Make status bar stick to bottom
        status_label = ttk.Label(status_frame, textvariable=self.status_str, anchor="w", relief=tk.SUNKEN)
        status_label.pack(fill=tk.X, expand=True, side=tk.BOTTOM)


        # --- Right Panel Widgets (Plot) ---
        plot_frame = ttk.LabelFrame(right_panel, text="Yield Locus Plot", padding="5")
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(5.5, 5.5))
        self.fig.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.92)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # --- Initial Plot ---
        self.plot_data() # Plot initial default/loaded data

    def parse_data(self, data_str):
        """Parses comma or space separated string data into a numpy array."""
        try:
            items = [item for item in data_str.replace(",", " ").split() if item]
            if not items: return np.array([])
            return np.array([float(item) for item in items])
        except ValueError:
            return None

    def browse_file(self):
        """Opens a file dialog to select a data file."""
        filepath = filedialog.askopenfilename(
            title="Select Experimental Data File",
            filetypes=[("Text files", "*.txt"), ("Data files", "*.dat"), ("All files", "*.*")]
        )
        if filepath: # If the user selected a file (didn't cancel)
            self.data_filepath_str.set(filepath)
            self.status_str.set(f"File selected: {os.path.basename(filepath)}. Click 'Load Data'.")

    def load_data_action(self):
        """Loads data from the file specified in data_filepath_str."""
        filepath = self.data_filepath_str.get()
        if not filepath:
            messagebox.showwarning("No File", "Please browse and select a data file first.")
            return

        try:
            s1_list = []
            s2_list = []
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'): # Skip empty lines and comments
                        continue
                    parts = line.split() # Split by whitespace
                    if len(parts) != 2:
                        raise ValueError(f"Line {line_num}: Expected 2 columns, found {len(parts)}")
                    try:
                        s1 = float(parts[0])
                        s2 = float(parts[1])
                        s1_list.append(s1)
                        s2_list.append(s2)
                    except ValueError:
                        raise ValueError(f"Line {line_num}: Could not convert '{parts[0]}' or '{parts[1]}' to number")

            if not s1_list: # Check if any data was actually read
                 messagebox.showwarning("Empty File", f"No valid data points found in '{os.path.basename(filepath)}'.")
                 self.status_str.set("Load failed: No data found in file.")
                 return

            # Update internal numpy arrays
            self.sigma1_exp_data = np.array(s1_list)
            self.sigma2_exp_data = np.array(s2_list)

            # Update the StringVars for display in Entry widgets (comma separated)
            self.sigma1_exp_str.set(", ".join(map(str, self.sigma1_exp_data)))
            self.sigma2_exp_str.set(", ".join(map(str, self.sigma2_exp_data)))

            self.status_str.set(f"Loaded {len(s1_list)} points from '{os.path.basename(filepath)}'. Ready to Optimize.")
            self.plot_data() # Update plot with newly loaded data

        except FileNotFoundError:
            messagebox.showerror("File Not Found", f"Error: The file '{filepath}' was not found.")
            self.status_str.set("Load failed: File not found.")
        except ValueError as ve:
            messagebox.showerror("File Format Error", f"Error reading file '{os.path.basename(filepath)}':\n{ve}")
            self.status_str.set("Load failed: Invalid file format.")
        except Exception as e:
            messagebox.showerror("Load Error", f"An unexpected error occurred while loading the file:\n{e}")
            self.status_str.set(f"Load failed: {e}")


    def run_optimization(self):
        """Get data (from Entry widgets), run optimization, update results and plot."""
        self.status_str.set("Preparing optimization...")
        self.master.update_idletasks()

        # --- Get Data from Entry Widgets (which might have been loaded/edited) ---
        sigma1_str = self.sigma1_exp_str.get()
        sigma2_str = self.sigma2_exp_str.get()
        sigma1_exp = self.parse_data(sigma1_str) # Re-parse from the text entry
        sigma2_exp = self.parse_data(sigma2_str) # Re-parse from the text entry

        # Store the parsed data back into the internal arrays
        self.sigma1_exp_data = sigma1_exp
        self.sigma2_exp_data = sigma2_exp

        # --- Validate Data ---
        if self.sigma1_exp_data is None or self.sigma2_exp_data is None:
            messagebox.showerror("Data Error", "Invalid format in current σ₁ or σ₂ data fields.")
            self.status_str.set("Error: Invalid data format in fields.")
            return
        if len(self.sigma1_exp_data) != len(self.sigma2_exp_data):
            messagebox.showerror("Data Error", "Current σ₁ and σ₂ data must have the same number of points.")
            self.status_str.set("Error: Mismatched data lengths in fields.")
            return
        if len(self.sigma1_exp_data) == 0:
             messagebox.showerror("Data Error", "No experimental data points entered or loaded.")
             self.status_str.set("Error: No data available for optimization.")
             return

        # --- Get Initial Guess and Bounds (same as before) ---
        try:
            alpha_init = float(self.alpha_init_str.get())
            beta_init = float(self.beta_init_str.get())
            k_init = float(self.k_init_str.get())
            x0 = np.array([alpha_init, beta_init, k_init])
        except ValueError:
            messagebox.showerror("Input Error", "Invalid format for initial guess parameters.")
            self.status_str.set("Error: Invalid initial guess.")
            return

        try:
            k_lower_bound = float(self.k_lower_bound_str.get())
            if k_lower_bound <= 0:
                 k_lower_bound = 1e-9
                 self.k_lower_bound_str.set(f"{k_lower_bound:.1e}")
                 messagebox.showwarning("Input Warning", f"k lower bound must be > 0. Set to {k_lower_bound:.1e}.")
            bounds = ([-np.inf, -np.inf, k_lower_bound], [np.inf, np.inf, np.inf])
        except ValueError:
            messagebox.showerror("Input Error", "Invalid format for k lower bound.")
            self.status_str.set("Error: Invalid k lower bound.")
            return

        if x0[2] <= bounds[0][2]:
             x0[2] = bounds[0][2] * 1.1
             self.k_init_str.set(f"{x0[2]:.4f}")
             messagebox.showwarning("Input Warning", f"Initial k₀ was below lower bound. Adjusted to {x0[2]:.4f}.")

        # --- Run Optimization (using the validated internal data arrays) ---
        self.status_str.set("Running optimization...")
        self.master.update_idletasks()
        try:
            result = least_squares(
                residuals,
                x0,
                args=(self.sigma1_exp_data, self.sigma2_exp_data), # Use internal arrays
                bounds=bounds,
                method='trf',
                ftol=1e-8, xtol=1e-8, gtol=1e-8
            )

            if result.success:
                alpha_opt, beta_opt, k_opt = result.x
                self.alpha_opt_str.set(f"{alpha_opt:.6f}")
                self.beta_opt_str.set(f"{beta_opt:.6f}")
                self.k_opt_str.set(f"{k_opt:.6f}")
                self.cost_str.set(f"{result.cost:.4e}")
                self.status_str.set(f"Optimization successful (Status: {result.status})")
                self.plot_data(alpha_opt, beta_opt, k_opt) # Pass optimized params to plot

            else:
                 # Handle failure (same as before)
                 messagebox.showwarning("Optimization Failed", f"Optimization did not converge: {result.message} (Status: {result.status})")
                 self.status_str.set(f"Optimization failed: {result.message}")
                 self.alpha_opt_str.set("Failed")
                 self.beta_opt_str.set("Failed")
                 self.k_opt_str.set("Failed")
                 self.cost_str.set("N/A")
                 self.plot_data() # Plot just the data


        except Exception as e:
            # Handle exceptions (same as before)
            messagebox.showerror("Optimization Error", f"An error occurred during optimization:\n{e}")
            self.status_str.set(f"Error: {e}")
            self.alpha_opt_str.set("Error")
            self.beta_opt_str.set("Error")
            self.k_opt_str.set("Error")
            self.cost_str.set("N/A")
            self.plot_data() # Plot just the data


    def plot_data(self, alpha=None, beta=None, k=None):
        """
        Updates the matplotlib plot using the internal data arrays
        (self.sigma1_exp_data, self.sigma2_exp_data).
        """
        self.ax.clear()

        # Use the internal numpy arrays for plotting
        sigma1_exp = self.sigma1_exp_data
        sigma2_exp = self.sigma2_exp_data

        plot_experimental = False
        lim_min, lim_max = -20, 20 # Default plot limits
        # Check if data is valid numpy arrays before plotting
        if isinstance(sigma1_exp, np.ndarray) and isinstance(sigma2_exp, np.ndarray) \
           and sigma1_exp.size > 0 and sigma1_exp.size == sigma2_exp.size:

             self.ax.scatter(sigma1_exp, sigma2_exp, color='red', zorder=5, label='Experimental Data')
             plot_experimental = True

             # Determine plot range based on data + k (same logic as before)
             s_data_min = min(np.min(sigma1_exp), np.min(sigma2_exp))
             s_data_max = max(np.max(sigma1_exp), np.max(sigma2_exp))
             s_abs_max = max(abs(s_data_min), abs(s_data_max))
             s_range_est = k * 1.5 if k is not None else s_abs_max
             plot_limit = max(s_abs_max, s_range_est) * 1.2
             lim_min = -plot_limit
             lim_max = plot_limit

        # Plot yield locus if optimized parameters are provided (same logic as before)
        plot_locus = False
        if alpha is not None and beta is not None and k is not None:
            try:
                sigma_range = np.linspace(lim_min, lim_max, 400)
                S1, S2 = np.meshgrid(sigma_range, sigma_range)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    Z = yield_func(S1, S2, alpha, beta, k)
                Z = np.nan_to_num(Z, nan=np.inf)
                if np.any(np.isfinite(Z)):
                    self.ax.contour(S1, S2, Z, levels=[0], colors='blue', linewidths=1.5)
                    plot_locus = True
                else:
                    print("Warning: Z contains no finite values. Cannot plot contour.")
                    # Update status only if optimization was reported successful previously
                    if self.status_str.get().startswith("Optimization successful"):
                         self.status_str.set(self.status_str.get() + " (Warning: Contour plot error)")
            except Exception as e:
                 print(f"Warning: Could not plot yield locus contour. Error: {e}")
                 if self.status_str.get().startswith("Optimization successful"):
                         self.status_str.set(self.status_str.get() + f" (Warning: Contour error - {e})")

        # Configure plot (same logic as before)
        self.ax.set_xlabel(r'$\sigma_1$ (MPa or units)')
        self.ax.set_ylabel(r'$\sigma_2$ (MPa or units)')
        title = 'Yield Locus'
        if plot_locus and plot_experimental: title = 'Optimized Yield Locus and Experimental Data'
        elif plot_experimental: title = 'Current Experimental Data'
        self.ax.set_title(title)
        if plot_experimental: self.ax.legend()
        self.ax.grid(linestyle=':', linewidth=0.5)
        self.ax.axis('equal')
        self.ax.set_xlim(lim_min, lim_max)
        self.ax.set_ylim(lim_min, lim_max)
        self.canvas.draw()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = YieldLocusOptimizerApp(root)
    root.mainloop()
