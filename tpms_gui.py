import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

# Try to import scikit-image for smooth meshing
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

class TPMSUtility:
    def __init__(self, root):
        self.root = root
        self.root.title("TPMS Utility v24.0")
        self.root.geometry("1200x950")
        
        # --- CLEAN EXIT FIX ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        if not HAS_SKIMAGE:
            messagebox.showwarning("Dependency Missing", "Please install 'scikit-image' (pip install scikit-image) for smooth previews.\nFalling back to voxel mode.")

        # --- Variables ---
        self.tpms_type = tk.StringVar(value="Gyroid")
        self.boundary = tk.StringVar(value="Box")
        self.export_mode = tk.StringVar(value="Watertight Solid")
        
        self.dim_x = tk.DoubleVar(value=30.0)
        self.dim_y = tk.DoubleVar(value=30.0)
        self.dim_z = tk.DoubleVar(value=30.0)
        self.cell_size = tk.DoubleVar(value=10.0)
        self.thickness = tk.DoubleVar(value=1.0)
        
        # This slider controls EXPORT resolution
        self.resolution = tk.IntVar(value=60)
        self.smooth_iters = tk.IntVar(value=5)

        self.res_feedback = tk.StringVar(value="Checking...")

        self.setup_ui()
        self.check_resolution_safety()

    def on_closing(self):
        try:
            plt.close('all')
            self.root.quit()
            self.root.destroy()
        except: pass
        finally: sys.exit(0)

    def setup_ui(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel ---
        left = ttk.Frame(self.paned, padding="15"); self.paned.add(left, weight=1)
        
        # 1. Geometry
        cfg = ttk.LabelFrame(left, text="Geometry Control", padding="10"); cfg.pack(fill=tk.X, pady=5)
        ttk.Label(cfg, text="Type:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(cfg, textvariable=self.tpms_type, values=("Gyroid", "I-WP", "Diamond"), state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Label(cfg, text="Boundary:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(cfg, textvariable=self.boundary, values=("Box", "Cylinder"), state="readonly").grid(row=1, column=1, sticky="ew")

        # 2. Dimensions
        params = ttk.LabelFrame(left, text="Dimensions (mm)", padding="10"); params.pack(fill=tk.X, pady=5)
        
        for v in [self.dim_x, self.dim_y, self.dim_z, self.thickness, self.resolution]:
            v.trace_add("write", lambda *args: self.check_resolution_safety())

        for i, (l, v) in enumerate([("X / Diam:", self.dim_x), ("Y:", self.dim_y), ("Z / Height:", self.dim_z), ("Unit Cell:", self.cell_size), ("Thickness:", self.thickness)]):
            ttk.Label(params, text=l).grid(row=i, column=0, sticky="w")
            ttk.Entry(params, textvariable=v).grid(row=i, column=1, sticky="ew")

        # 3. Quality & Safety
        quality = ttk.LabelFrame(left, text="Mesh Quality & Safety", padding="10"); quality.pack(fill=tk.X, pady=5)
        
        ttk.Label(quality, text="Export Resolution (Voxels):").pack(anchor="w")
        scale = tk.Scale(quality, from_=10, to=200, variable=self.resolution, orient="horizontal")
        scale.pack(fill=tk.X)
        
        self.lbl_safe = ttk.Label(quality, textvariable=self.res_feedback, foreground="red", wraplength=250)
        self.lbl_safe.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Button(quality, text="[AUTO] Fix Export Resolution", command=self.set_safe_res).pack(fill=tk.X, pady=2)

        ttk.Label(quality, text="Smoothing Iterations:").pack(anchor="w", pady=(5,0))
        tk.Scale(quality, from_=0, to=50, variable=self.smooth_iters, orient="horizontal").pack(fill=tk.X)
        
        # --- THE BUTTON ---
        ttk.Button(left, text=">> Update Preview", command=self.refresh_3d_preview).pack(fill=tk.X, pady=10)

        # 4. Export
        export = ttk.LabelFrame(left, text="Export Options", padding="10"); export.pack(fill=tk.X, pady=5)
        ttk.Combobox(export, textvariable=self.export_mode, values=("Watertight Solid", "Double-Sided (Open)", "Zero-Thickness Surface"), state="readonly").pack(fill=tk.X, pady=5)
        
        ttk.Button(export, text="Export OBJ", command=lambda: self.process("obj")).pack(fill=tk.X, pady=2)
        ttk.Button(export, text="Export STL", command=lambda: self.process("stl")).pack(fill=tk.X, pady=2)

        self.status = ttk.Label(left, text="Ready (Click 'Update Preview')", foreground="#2ecc71"); self.status.pack(pady=10)

        # --- Right Panel ---
        right = ttk.Frame(self.paned, padding="5"); self.paned.add(right, weight=2)
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.text2D(0.5, 0.5, "Click 'Update Preview'", transform=self.ax.transAxes, ha="center")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def check_resolution_safety(self):
        try:
            max_dim = max(self.dim_x.get(), self.dim_y.get(), self.dim_z.get())
            thk = self.thickness.get()
            res = self.resolution.get()
            
            if thk <= 0: return 
            
            voxel_size = max_dim / res
            voxels_per_wall = thk / voxel_size
            
            if voxels_per_wall < 1.0:
                self.lbl_safe.config(text=f"[CRITICAL] Export Wall < 1 voxel ({voxels_per_wall:.1f}). Holes guaranteed.", foreground="red")
            elif voxels_per_wall < 2.0:
                self.lbl_safe.config(text=f"[RISKY] Export Wall {voxels_per_wall:.1f} voxels. Gaps possible.", foreground="#d35400")
            elif voxels_per_wall < 3.0:
                self.lbl_safe.config(text=f"[OKAY] Export Wall {voxels_per_wall:.1f} voxels.", foreground="#f39c12")
            else:
                self.lbl_safe.config(text=f"[SAFE] Export Wall {voxels_per_wall:.1f} voxels.", foreground="#27ae60")
        except: pass

    def set_safe_res(self):
        try:
            max_dim = max(self.dim_x.get(), self.dim_y.get(), self.dim_z.get())
            thk = self.thickness.get()
            if thk > 0:
                safe_res = int(np.ceil(max_dim / (thk / 3.0)))
                self.resolution.set(max(safe_res, 40))
        except: pass

    def get_field_and_grad(self, pts):
        scale = 2 * np.pi / max(0.1, self.cell_size.get())
        t_pts = pts * scale
        x, y, z = t_pts[:,0], t_pts[:,1], t_pts[:,2]
        t = self.tpms_type.get()
        
        if t == "Gyroid":
            val = np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x); gc = 0.6
            dx, dy, dz = np.cos(x)*np.cos(y)-np.sin(z)*np.sin(x), np.cos(y)*np.cos(z)-np.sin(x)*np.sin(y), np.cos(z)*np.cos(x)-np.sin(y)*np.sin(z)
        elif t == "I-WP":
            val = 2*(np.cos(x)*np.cos(y) + np.cos(y)*np.cos(z) + np.cos(z)*np.cos(x)) - (np.cos(2*x) + np.cos(2*y) + np.cos(2*z)); gc = 0.3
            dx, dy, dz = -2*np.sin(x)*(np.cos(y)+np.cos(z)) + 2*np.sin(2*x), -2*np.sin(y)*(np.cos(x)+np.cos(z)) + 2*np.sin(2*y), -2*np.sin(z)*(np.cos(x)+np.cos(y)) + 2*np.sin(2*z)
        else: # Diamond
            val = np.sin(x)*np.sin(y)*np.sin(z) + np.sin(x)*np.cos(y)*np.cos(z) + np.cos(x)*np.sin(y)*np.cos(z) + np.cos(x)*np.cos(y)*np.sin(z); gc = 0.5
            dx = np.cos(x)*(np.sin(y)*np.sin(z)+np.cos(y)*np.cos(z)) - np.sin(x)*(np.sin(y)*np.cos(z)+np.cos(y)*np.sin(z))
            dy = np.cos(y)*(np.sin(x)*np.sin(z)+np.cos(x)*np.cos(z)) - np.sin(y)*(np.sin(x)*np.cos(z)+np.cos(x)*np.sin(z))
            dz = np.cos(z)*(np.sin(x)*np.sin(y)+np.cos(x)*np.cos(y)) - np.sin(z)*(np.sin(x)*np.cos(y)+np.cos(x)*np.sin(y))

        mode = self.export_mode.get()
        if mode == "Zero-Thickness Surface":
            field = -val
            grad = -np.stack([dx, dy, dz], axis=1) * scale
        else:
            t_iso = (self.thickness.get() / max(0.1, self.cell_size.get())) * (2 * np.pi * gc)
            field = t_iso - np.abs(val)
            grad = -np.sign(val)[:, np.newaxis] * np.stack([dx, dy, dz], axis=1) * scale
        
        if self.boundary.get() == "Cylinder":
            field[pts[:,0]**2 + pts[:,1]**2 > (self.dim_x.get()/2)**2] = -1.0
            
        return field, grad

    def refresh_3d_preview(self):
        self.status.config(text="Computing Mesh...", foreground="blue")
        self.root.update()
        
        L, W, H = self.dim_x.get(), self.dim_y.get(), self.dim_z.get()
        thk = self.thickness.get()
        
        # --- ADAPTIVE RESOLUTION LOGIC ---
        # We need at least 2 voxels to cover the wall thickness to avoid holes
        # Max dimension divided by (thickness / 2.0) gives us the safe resolution
        if thk > 0:
            calc_res = int(max(L, W, H) / (thk / 2.2))
        else:
            calc_res = 32
            
        # Clamp resolution: Min 30 (for shape), Max 90 (for matplotlib performance)
        p_res = max(30, min(90, calc_res))
        
        x = np.linspace(-L/2, L/2, p_res); y = np.linspace(-W/2, W/2, p_res); z = np.linspace(-H/2, H/2, p_res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        f, _ = self.get_field_and_grad(pts)
        f_grid = f.reshape(p_res, p_res, p_res)

        self.ax.clear()

        if HAS_SKIMAGE:
            try:
                verts, faces, _, _ = measure.marching_cubes(f_grid, level=0)
                
                # Rescale
                verts[:, 0] = verts[:, 0] * (L / (p_res-1)) - L/2
                verts[:, 1] = verts[:, 1] * (W / (p_res-1)) - W/2
                verts[:, 2] = verts[:, 2] * (H / (p_res-1)) - H/2
                
                z_vals = verts[faces].mean(axis=1)[:, 2]
                norm = Normalize(vmin=-H/2, vmax=H/2)
                colors = cm.viridis(norm(z_vals))
                
                # Add lighting to make holes/depth easier to distinguish
                ls = LightSource(azdeg=225, altdeg=45)
                # Note: Matplotlib's shade=True with facecolors can be tricky, 
                # so we stick to simple facecolors + edgecolors for clarity
                
                mesh = Poly3DCollection(verts[faces], facecolors=colors, alpha=1.0, 
                                        edgecolors='black', linewidths=0.1)
                self.ax.add_collection3d(mesh)
                self.ax.set_title(f"Preview (Res: {p_res}x{p_res}x{p_res})")
            except:
                 self.status.config(text="Resolution Too Low for Thickness", foreground="red")
                 self.ax.text2D(0.5, 0.5, "Structure too thin for preview.\nIncrease Thickness or Dim.", transform=self.ax.transAxes, ha="center", color="red")
        else:
             self.ax.text(0,0,0, "Install scikit-image for smooth mode", c='red')

        self.ax.set_xlim(-L/2, L/2); self.ax.set_ylim(-W/2, W/2); self.ax.set_zlim(-H/2, H/2)
        self.ax.set_axis_off() 
        self.ax.view_init(elev=30, azim=45)
        self.canvas.draw()
        self.status.config(text="Ready", foreground="#2ecc71")

    def process(self, fmt):
        if not HAS_SKIMAGE:
            messagebox.showerror("Error", "Exporting smooth meshes requires scikit-image.")
            return

        try:
            self.status.config(text="Processing High-Res...", foreground="blue"); self.root.update()
            L, W, H = self.dim_x.get(), self.dim_y.get(), self.dim_z.get()
            res = self.resolution.get()
            
            x_v = np.linspace(-L/2, L/2, res); y_v = np.linspace(-W/2, W/2, res); z_v = np.linspace(-H/2, H/2, res)
            X, Y, Z = np.meshgrid(x_v, y_v, z_v, indexing='ij')
            
            f_vals, _ = self.get_field_and_grad(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1))
            f_grid = f_vals.reshape(res, res, res)
            
            if self.export_mode.get() == "Watertight Solid":
                f_grid = np.pad(f_grid, 1, mode='constant', constant_values=-1.0)
            
            verts, faces, _, _ = measure.marching_cubes(f_grid, level=0)
            
            scale_fac = [L/(res-1), W/(res-1), H/(res-1)]
            shift = [L/2, W/2, H/2]
            
            if self.export_mode.get() == "Watertight Solid":
                 shift = [s + scale_fac[i] for i, s in enumerate(shift)]
                 
            verts[:, 0] = verts[:, 0] * scale_fac[0] - shift[0]
            verts[:, 1] = verts[:, 1] * scale_fac[1] - shift[1]
            verts[:, 2] = verts[:, 2] * scale_fac[2] - shift[2]

            if self.smooth_iters.get() > 0:
                adj = [set() for _ in range(len(verts))]
                for f in faces:
                    adj[f[0]].add(f[1]); adj[f[0]].add(f[2])
                    adj[f[1]].add(f[0]); adj[f[1]].add(f[2])
                    adj[f[2]].add(f[0]); adj[f[2]].add(f[1])
                
                adj_list = [list(a) for a in adj]
                clamp_limit = self.thickness.get() * 0.4
                
                for _ in range(self.smooth_iters.get()):
                    relaxed = np.zeros_like(verts)
                    for idx, nbs in enumerate(adj_list):
                        if nbs: relaxed[idx] = verts[nbs].mean(axis=0)
                    
                    fv, fg = self.get_field_and_grad(relaxed)
                    ns = np.sum(fg**2, axis=1)
                    mask = ns > 1e-9
                    step = (fv[mask] / ns[mask])[:, np.newaxis] * fg[mask]
                    
                    step_mags = np.linalg.norm(step, axis=1)
                    clip_mask = step_mags > clamp_limit
                    if np.any(clip_mask):
                        step[clip_mask] *= (clamp_limit / step_mags[clip_mask])[:, np.newaxis]
                        
                    relaxed[mask] -= step
                    verts = relaxed

            path = filedialog.asksaveasfilename(defaultextension=f".{fmt}")
            if path:
                if fmt == "xyz": np.savetxt(path, verts, fmt="%.6f")
                elif fmt == "obj":
                    with open(path, 'w') as f:
                        for v in verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                        for fc in faces: f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
                elif fmt == "stl":
                    with open(path, 'w') as f:
                        f.write("solid TPMS\n")
                        for fc in faces:
                            v = verts[fc]
                            n = np.cross(v[1]-v[0], v[2]-v[0])
                            norm_len = np.linalg.norm(n)
                            if norm_len > 0: n /= norm_len
                            f.write(f"facet normal {n[0]} {n[1]} {n[2]}\nouter loop\n")
                            for k in range(3): f.write(f"vertex {v[k][0]} {v[k][1]} {v[k][2]}\n")
                            f.write("endloop\nendfacet\n")
                        f.write("endsolid TPMS\n")
            self.status.config(text="Ready", foreground="#2ecc71")
        except Exception as e: messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk(); app = TPMSUtility(root); root.mainloop()