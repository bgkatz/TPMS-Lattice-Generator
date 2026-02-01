import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

class TPMSAntiDimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TPMS Anti-Dimple Solid (v9.0)")
        self.root.geometry("480x750")

        self.tpms_type = tk.StringVar(value="Gyroid")
        self.dim_x = tk.DoubleVar(value=30.0); self.dim_y = tk.DoubleVar(value=30.0); self.dim_z = tk.DoubleVar(value=30.0)
        self.cell_size = tk.DoubleVar(value=10.0)
        self.resolution = tk.IntVar(value=40) # Increased default for thin walls
        self.thickness = tk.DoubleVar(value=1.5)
        self.smooth_iters = tk.IntVar(value=20)

        self.setup_ui()

    def setup_ui(self):
        main = ttk.Frame(self.root, padding="20")
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Lattice Geometry", font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Combobox(main, textvariable=self.tpms_type, values=("Gyroid", "I-WP", "Diamond", "Primitive"), state="readonly").pack(fill=tk.X, pady=5)

        dim_frame = ttk.LabelFrame(main, text="Dimensions (mm)", padding="10")
        dim_frame.pack(fill=tk.X, pady=10)
        for l, v in [("X:", self.dim_x), ("Y:", self.dim_y), ("Z:", self.dim_z)]:
            f = ttk.Frame(dim_frame); f.pack(fill=tk.X)
            ttk.Label(f, text=l, width=5).pack(side=tk.LEFT); ttk.Entry(f, textvariable=v).pack(side=tk.RIGHT, expand=True, fill=tk.X)

        param_frame = ttk.LabelFrame(main, text="Wall Properties", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        self.create_entry(param_frame, "Unit Cell (mm):", self.cell_size)
        self.create_entry(param_frame, "Thickness (mm):", self.thickness)
        self.create_entry(param_frame, "Resolution (40+ for thin):", self.resolution)
        
        ttk.Label(param_frame, text="Smoothness (Safe with Reprojection):", font=("Arial", 9)).pack(anchor="w", pady=(10,0))
        ttk.Scale(param_frame, from_=0, to=50, variable=self.smooth_iters, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Button(main, text="💾 Export Watertight Solid", command=self.generate).pack(fill=tk.X, pady=20, ipady=10)
        self.status = ttk.Label(main, text="Ready", foreground="grey"); self.status.pack()

    def create_entry(self, parent, label, var):
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label, width=20).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=var).pack(side=tk.RIGHT, expand=True, fill=tk.X)

    def get_field(self, x, y, z):
        # Helper to get field value at any point (not just grid points)
        scale = 2 * np.pi / self.cell_size.get()
        tx, ty, tz = x * scale, y * scale, z * scale
        tpms = self.tpms_type.get()
        if tpms == "Gyroid": vol = np.sin(tx)*np.cos(ty) + np.sin(ty)*np.cos(tz) + np.sin(tz)*np.cos(tx); g_corr = 0.6
        elif tpms == "I-WP": vol = 2*(np.cos(tx)*np.cos(ty) + np.cos(ty)*np.cos(tz) + np.cos(tx)*np.cos(tz)) - (np.cos(2*tx) + np.cos(2*ty) + np.cos(2*tz)); g_corr = 0.3
        elif tpms == "Diamond": vol = np.sin(tx)*np.sin(ty)*np.sin(tz) + np.sin(tx)*np.cos(ty)*np.cos(tz) + np.cos(tx)*np.sin(ty)*np.cos(tz) + np.cos(tx)*np.cos(ty)*np.sin(tz); g_corr = 0.5
        else: vol = np.cos(tx) + np.cos(ty) + np.cos(tz); g_corr = 0.5
        
        t_iso = (self.thickness.get() / self.cell_size.get()) * (2 * np.pi * g_corr)
        return t_iso - np.abs(vol)

    def generate(self):
        try:
            self.status.config(text="Calculating Field...")
            self.root.update()

            L, W, H = self.dim_x.get(), self.dim_y.get(), self.dim_z.get()
            res = self.resolution.get()
            x_v = np.linspace(-L/2, L/2, res); y_v = np.linspace(-W/2, W/2, res); z_v = np.linspace(-H/2, H/2, res)
            X, Y, Z = np.meshgrid(x_v, y_v, z_v, indexing='ij')

            # Calculate the field on the grid
            field = self.get_field(X, Y, Z)
            padded_field = np.pad(field, 1, mode='constant', constant_values=-1.0)
            
            # Padded coordinates
            dx = x_v[1]-x_v[0]
            px = np.linspace(x_v[0]-dx, x_v[-1]+dx, res+2)
            py = np.linspace(y_v[0]-dx, y_v[-1]+dx, res+2)
            pz = np.linspace(z_v[0]-dx, z_v[-1]+dx, res+2)
            pres = res + 2

            verts, faces, voxel_to_idx = [], [], {}

            # Surface Nets
            for i in range(pres-1):
                for j in range(pres-1):
                    for k in range(pres-1):
                        corners = padded_field[i:i+2, j:j+2, k:k+2]
                        if not (np.all(corners > 0) or np.all(corners < 0)):
                            v_pos = np.array([(px[i]+px[i+1])/2, (py[j]+py[j+1])/2, (pz[k]+pz[k+1])/2])
                            voxel_to_idx[(i, j, k)] = len(verts); verts.append(v_pos)

            for i in range(pres):
                for j in range(pres):
                    for k in range(pres):
                        if i < pres-1 and (padded_field[i,j,k] > 0) != (padded_field[i+1,j,k] > 0):
                            v = [(i,j-1,k-1), (i,j,k-1), (i,j,k), (i,j-1,k)]
                            if all(p in voxel_to_idx for p in v): 
                                f = [voxel_to_idx[p] for p in v]
                                faces.append(f if padded_field[i,j,k] > 0 else f[::-1])
                        if j < pres-1 and (padded_field[i,j,k] > 0) != (padded_field[i,j+1,k] > 0):
                            v = [(i-1,j,k-1), (i,j,k-1), (i,j,k), (i-1,j,k)]
                            if all(p in voxel_to_idx for p in v): 
                                f = [voxel_to_idx[p] for p in v]
                                faces.append(f if padded_field[i,j,k] < 0 else f[::-1])
                        if k < pres-1 and (padded_field[i,j,k] > 0) != (padded_field[i,j,k+1] > 0):
                            v = [(i-1,j-1,k), (i,j-1,k), (i,j,k), (i-1,j,k)]
                            if all(p in voxel_to_idx for p in v): 
                                f = [voxel_to_idx[p] for p in v]
                                faces.append(f if padded_field[i,j,k] > 0 else f[::-1])

            # Constrained Smoothing Loop
            verts = np.array(verts)
            iters = self.smooth_iters.get()
            if iters > 0:
                self.status.config(text="Constrained Smoothing...")
                self.root.update()
                adj = [set() for _ in range(len(verts))]
                for f in faces:
                    for idx in range(4): adj[f[idx]].add(f[(idx+1)%4]); adj[f[idx]].add(f[(idx-1)%4])
                
                eps = 1e-4
                for _ in range(iters):
                    new_v = verts.copy()
                    for idx, nbs in enumerate(adj):
                        if not nbs: continue
                        # 1. Laplacian Step
                        relaxed = verts[list(nbs)].mean(axis=0)
                        
                        # 2. Reprojection Step (Newton's Method)
                        # Push relaxed vertex back to field=0 to maintain thickness
                        curr_v = relaxed
                        for _ in range(2): # 2 Newton steps is usually enough
                            val = self.get_field(curr_v[0], curr_v[1], curr_v[2])
                            # Finite difference gradient
                            gx = (self.get_field(curr_v[0]+eps, curr_v[1], curr_v[2]) - val)/eps
                            gy = (self.get_field(curr_v[0], curr_v[1]+eps, curr_v[2]) - val)/eps
                            gz = (self.get_field(curr_v[0], curr_v[1], curr_v[2]+eps) - val)/eps
                            grad = np.array([gx, gy, gz])
                            norm_sq = np.dot(grad, grad)
                            if norm_sq > 1e-9:
                                curr_v = curr_v - (val / norm_sq) * grad
                        new_v[idx] = curr_v
                    verts = new_v

            # Export
            path = filedialog.asksaveasfilename(defaultextension=".obj", filetypes=[("OBJ", "*.obj")])
            if path:
                with open(path, 'w') as f:
                    for v in verts: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                    for fc in faces: f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1} {fc[3]+1}\n")
                messagebox.showinfo("Success", "Solid Quad-Mesh Exported.")
            self.status.config(text="Ready")
        except Exception as e: messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk(); app = TPMSAntiDimpleApp(root); root.mainloop()