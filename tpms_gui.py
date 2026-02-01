import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from skimage.measure import marching_cubes
import trimesh
import pandas as pd
import os

class TPMSMasterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TPMS Master Generator v3.0")
        self.root.geometry("500x850")

        # --- Variables ---
        self.tpms_type = tk.StringVar(value="Gyroid")
        self.shape_type = tk.StringVar(value="Box")
        
        # Dimensions
        self.dim_x = tk.DoubleVar(value=50.0); self.dim_y = tk.DoubleVar(value=50.0); self.dim_z = tk.DoubleVar(value=50.0)
        self.radius = tk.DoubleVar(value=25.0); self.height = tk.DoubleVar(value=50.0)
        
        # Lattice Params
        self.cell_size = tk.DoubleVar(value=10.0)
        self.resolution = tk.IntVar(value=70)
        self.thickness = tk.DoubleVar(value=0.0)
        self.sample_rate = tk.IntVar(value=5)

        self.setup_ui()

    def setup_ui(self):
        main_scroll = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_scroll.yview)
        self.scrollable_frame = ttk.Frame(main_scroll, padding="20")

        self.scrollable_frame.bind("<Configure>", lambda e: main_scroll.configure(scrollregion=main_scroll.bbox("all")))
        main_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_scroll.configure(yscrollcommand=scrollbar.set)

        main_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. Type Selection
        self.section_label("1. Geometry Selection")
        ttk.Label(self.scrollable_frame, text="TPMS Type:").pack(anchor="w")
        ttk.Combobox(self.scrollable_frame, textvariable=self.tpms_type, values=("Gyroid", "I-WP", "Diamond", "Primitive", "Lidinoid"), state="readonly").pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.scrollable_frame, text="Boundary:").pack(anchor="w")
        shape_cb = ttk.Combobox(self.scrollable_frame, textvariable=self.shape_type, values=("Box", "Cylinder"), state="readonly")
        shape_cb.bind("<<ComboboxSelected>>", self.toggle_shape)
        shape_cb.pack(fill=tk.X, pady=(0, 10))

        # 2. Dimensions
        self.dim_frame = ttk.LabelFrame(self.scrollable_frame, text="Dimensions (mm)", padding="10")
        self.dim_frame.pack(fill=tk.X, pady=5)
        
        self.box_ui = ttk.Frame(self.dim_frame)
        for l, v in [("X:", self.dim_x), ("Y:", self.dim_y), ("Z:", self.dim_z)]:
            f = ttk.Frame(self.box_ui); f.pack(fill=tk.X)
            ttk.Label(f, text=l, width=5).pack(side=tk.LEFT); ttk.Entry(f, textvariable=v).pack(side=tk.RIGHT, expand=True, fill=tk.X)
        self.box_ui.pack(fill=tk.X)

        self.cyl_ui = ttk.Frame(self.dim_frame)
        for l, v in [("Radius:", self.radius), ("Height:", self.height)]:
            f = ttk.Frame(self.cyl_ui); f.pack(fill=tk.X)
            ttk.Label(f, text=l, width=10).pack(side=tk.LEFT); ttk.Entry(f, textvariable=v).pack(side=tk.RIGHT, expand=True, fill=tk.X)

        # 3. Lattice Settings
        self.section_label("2. Lattice Settings")
        self.create_entry(self.scrollable_frame, "Unit Cell (mm):", self.cell_size)
        self.create_entry(self.scrollable_frame, "Resolution:", self.resolution)
        self.create_entry(self.scrollable_frame, "Thickness (mm):", self.thickness)
        ttk.Label(self.scrollable_frame, text="(0 = Sheet Mesh)", font=("Arial", 8, "italic")).pack(anchor="w")

        # 4. Point Cloud Settings
        self.section_label("3. Point Cloud Settings")
        ttk.Label(self.scrollable_frame, text="Downsample (every Nth point):").pack(anchor="w")
        ttk.Scale(self.scrollable_frame, from_=1, to=50, variable=self.sample_rate, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(self.scrollable_frame, textvariable=self.sample_rate).pack()

        # 5. Buttons
        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', pady=15)
        ttk.Button(self.scrollable_frame, text="👁 Preview Mesh", command=self.preview).pack(fill=tk.X, pady=2)
        ttk.Button(self.scrollable_frame, text="💾 Export STL / OBJ", command=self.export_mesh).pack(fill=tk.X, pady=2)
        ttk.Button(self.scrollable_frame, text="☁️ Export Point Cloud (CSV)", command=self.export_points).pack(fill=tk.X, pady=2)
        
        self.status = ttk.Label(self.scrollable_frame, text="Ready", foreground="blue")
        self.status.pack(pady=10)

    def section_label(self, text):
        ttk.Label(self.scrollable_frame, text=text, font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(15, 5))

    def create_entry(self, parent, label, var):
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label, width=15).pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=var).pack(side=tk.RIGHT, expand=True, fill=tk.X)

    def toggle_shape(self, event=None):
        if self.shape_type.get() == "Box":
            self.cyl_ui.pack_forget(); self.box_ui.pack(fill=tk.X)
        else:
            self.box_ui.pack_forget(); self.cyl_ui.pack(fill=tk.X)

    def generate_data(self, for_points=False):
        L, W, H = (self.dim_x.get(), self.dim_y.get(), self.dim_z.get()) if self.shape_type.get() == "Box" else (self.radius.get()*2, self.radius.get()*2, self.height.get())
        res = self.resolution.get()
        x_vals = np.linspace(-L/2, L/2, res); y_vals = np.linspace(-W/2, W/2, res); z_vals = np.linspace(-H/2, H/2, res)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

        scale = 2 * np.pi / self.cell_size.get()
        x, y, z = X * scale, Y * scale, Z * scale
        tpms = self.tpms_type.get()
        
        # Standard equations
        if tpms == "Gyroid": vol = np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x); grad = 0.6
        elif tpms == "I-WP": vol = 2*(np.cos(x)*np.cos(y) + np.cos(y)*np.cos(z) + np.cos(z)*np.cos(x)) - (np.cos(2*x) + np.cos(2*y) + np.cos(2*z)); grad = 0.3
        elif tpms == "Diamond": vol = np.sin(x)*np.sin(y)*np.sin(z) + np.sin(x)*np.cos(y)*np.cos(z) + np.cos(x)*np.sin(y)*np.cos(z) + np.cos(x)*np.cos(y)*np.sin(z); grad = 0.5
        elif tpms == "Primitive": vol = np.cos(x) + np.cos(y) + np.cos(z); grad = 0.5
        else: vol = 0.5*(np.sin(2*x)*np.cos(y)*np.sin(z) + np.sin(2*y)*np.cos(z)*np.sin(x) + np.sin(2*z)*np.cos(x)*np.sin(y)) - 0.5*(np.cos(2*x)*np.cos(2*y) + np.cos(2*y)*np.cos(2*z) + np.cos(2*z)*np.cos(2*x)) + 0.15; grad = 0.4

        t_mm = self.thickness.get()
        # If exporting points, we always use the mid-surface (t=0)
        if for_points or t_mm <= 0:
            verts, faces, _, _ = marching_cubes(vol, level=0.0)
        else:
            t_iso = (t_mm / self.cell_size.get()) * (2 * np.pi * grad)
            verts, faces, _, _ = marching_cubes(t_iso - np.abs(vol), level=0.0)

        verts[:, 0] = (verts[:, 0] / (res-1)) * L - L/2
        verts[:, 1] = (verts[:, 1] / (res-1)) * W - W/2
        verts[:, 2] = (verts[:, 2] / (res-1)) * H - H/2

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if self.shape_type.get() == "Cylinder":
            radii = np.sqrt(mesh.vertices[:, 0]**2 + mesh.vertices[:, 1]**2)
            face_mask = (radii[mesh.faces] < self.radius.get()).all(axis=1)
            mesh.update_faces(face_mask); mesh.remove_unreferenced_vertices()
        
        return mesh

    def preview(self):
        try:
            self.status.config(text="Generating Mesh...")
            self.root.update()
            mesh = self.generate_data()
            mesh.show()
            self.status.config(text="Ready")
        except Exception as e: messagebox.showerror("Error", str(e))

    def export_mesh(self):
        try:
            mesh = self.generate_data()
            path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL", "*.stl"), ("OBJ", "*.obj")])
            if path:
                mesh.export(path)
                messagebox.showinfo("Success", "Mesh Exported")
        except Exception as e: messagebox.showerror("Error", str(e))

    def export_points(self):
        try:
            self.status.config(text="Extracting Points...")
            self.root.update()
            mesh = self.generate_data(for_points=True)
            pts = mesh.vertices[::self.sample_rate.get()]
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                pd.DataFrame(pts, columns=['X','Y','Z']).to_csv(path, index=False)
                messagebox.showinfo("Success", f"Saved {len(pts)} points.")
            self.status.config(text="Ready")
        except Exception as e: messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk(); app = TPMSMasterApp(root); root.mainloop()