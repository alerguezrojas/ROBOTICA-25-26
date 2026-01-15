import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from mpl_toolkits.mplot3d import Axes3D
import time

# --- CLASE 1: MOTOR MATEMÁTICO ---
class RotationEngine:
    @staticmethod
    def get_rotation_matrices(roll, pitch, yaw):
        """Calcula matrices elementales y la cadena jerárquica Z-Y-X."""
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)

        # Matrices de rotación elementales
        Rx = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
        Ry = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
        Rz = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
        
        # Jerarquía para anillos físicos
        return Rz, (Rz @ Ry), (Rz @ Ry @ Rx)

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """Conversión a Cuaternión de Hamilton."""
        cr, sr = np.cos(roll*0.5), np.sin(roll*0.5)
        cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
        cy, sy = np.cos(yaw*0.5), np.sin(yaw*0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])

    @staticmethod
    def quaternion_to_matrix(q):
        """Matriz de rotación desde cuaternión."""
        norm = np.linalg.norm(q)
        if norm > 0: q = q / norm
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])

    @staticmethod
    def slerp(q1, q2, t):
        """Interpolación Lineal Esférica (Shortest Path)."""
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        
        dot = np.clip(dot, -1.0, 1.0)
        theta_0 = np.arccos(dot)
        if theta_0 < 1e-6: return q1
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q1) + (s1 * q2)

    @staticmethod
    def euler_lerp(ang1, ang2, t):
        """Interpolación lineal simple de ángulos (Euler)."""
        return ang1 + t * (ang2 - ang1)

# --- CLASE 2: INTERFAZ Y VISUALIZACIÓN ---
class GimbalApp:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        plt.subplots_adjust(left=0.25, bottom=0.22, right=0.95, top=0.95)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Estado de puntos de trayectoria
        self.q_start, self.rpy_start = None, None
        self.q_end, self.rpy_end = None, None
        
        # Geometría
        self.res = 100
        self.theta_linspace = np.linspace(0, 2*np.pi, self.res)
        self.ring_rads = [1.2, 1.0, 0.8]
        
        self.setup_ui()
        self.update(0)

    def draw_grids(self):
        limit = 1.5
        vals = np.linspace(-limit, limit, 9)
        for v in vals:
            self.ax.plot([-limit, limit], [v, v], [-limit, -limit], color='#334155', alpha=0.1, lw=0.5)
            self.ax.plot([v, v], [-limit, limit], [-limit, -limit], color='#334155', alpha=0.1, lw=0.5)
            self.ax.plot([-limit, limit], [limit, limit], [v, v], color='#334155', alpha=0.05, lw=0.5)
            self.ax.plot([v, v], [limit, limit], [-limit, limit], color='#334155', alpha=0.05, lw=0.5)

    def draw_camera(self, matrix, base_color='#64748b', lens_color='#3b82f6', alpha=1.0):
        # Cuerpo
        box = np.array([
            [-0.3, -0.2, -0.2], [0.3, -0.2, -0.2], [0.3, 0.2, -0.2], [-0.3, 0.2, -0.2], [-0.3, -0.2, -0.2],
            [-0.3, -0.2, 0.2], [0.3, -0.2, 0.2], [0.3, 0.2, 0.2], [-0.3, 0.2, 0.2], [-0.3, -0.2, 0.2]
        ]).T
        # Lente
        lens = np.array([
            [0, 0, 0.2], [0.15, 0.15, 0.5], [-0.15, 0.15, 0.5], [0, 0, 0.2],
            [0.15, -0.15, 0.5], [-0.15, -0.15, 0.5], [0, 0, 0.2]
        ]).T
        for part, color in [(box, base_color), (lens, lens_color)]:
            r_part = matrix @ part
            self.ax.plot(r_part[0], r_part[1], r_part[2], color=color, linewidth=2, alpha=alpha)

    def draw_basis(self, matrix, alpha=1.0):
        colors = ['#ef4444', '#22c55e', '#3b82f6']
        basis = np.eye(3)
        for i in range(3):
            vec = matrix @ basis[:, i]
            self.ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=colors[i], 
                          length=0.8, linewidth=3, arrow_length_ratio=0.1, alpha=alpha)

    def draw_ring(self, radius, matrix, color, plane='XY', alpha=0.3):
        c, s = np.cos(self.theta_linspace), np.sin(self.theta_linspace)
        z = np.zeros(self.res)
        if plane == 'XY': points = np.vstack((radius*c, radius*s, z))
        elif plane == 'XZ': points = np.vstack((radius*c, z, radius*s))
        elif plane == 'YZ': points = np.vstack((z, radius*c, radius*s))
        rotated = matrix @ points
        self.ax.plot(rotated[0], rotated[1], rotated[2], color=color, alpha=alpha, linewidth=2)

    def draw_trajectories(self):
        if self.q_start is not None and self.q_end is not None:
            t_vals = np.linspace(0, 1, 60)
            pts_slerp = []
            for t in t_vals:
                qt = RotationEngine.slerp(self.q_start, self.q_end, t)
                pts_slerp.append(RotationEngine.quaternion_to_matrix(qt) @ [0, 0, 0.8])
            pts_slerp = np.array(pts_slerp).T
            self.ax.plot(pts_slerp[0], pts_slerp[1], pts_slerp[2], color='white', linestyle='--', alpha=0.7, lw=1.5)

            pts_euler = []
            for t in t_vals:
                angs = RotationEngine.euler_lerp(self.rpy_start, self.rpy_end, t)
                _, _, M = RotationEngine.get_rotation_matrices(angs[0], angs[1], angs[2])
                pts_euler.append(M @ [0, 0, 0.8])
            pts_euler = np.array(pts_euler).T
            self.ax.plot(pts_euler[0], pts_euler[1], pts_euler[2], color='#fb923c', linestyle=':', alpha=0.7, lw=1.5)

    def setup_ui(self):
        # Sliders
        self.s_roll = Slider(plt.axes([0.35, 0.12, 0.45, 0.03]), 'Roll (X)', -180, 180, 0, color='#ec4899')
        self.s_pitch = Slider(plt.axes([0.35, 0.08, 0.45, 0.03]), 'Pitch (Y)', -90, 90, 0, color='#22c55e')
        self.s_yaw = Slider(plt.axes([0.35, 0.04, 0.45, 0.03]), 'Yaw (Z)', -180, 180, 0, color='#06b6d4')

        # Controles laterales
        self.radio = RadioButtons(plt.axes([0.02, 0.72, 0.18, 0.12]), ('Método Euler', 'Método Cuaternión'))
        self.btn_lock = Button(plt.axes([0.02, 0.64, 0.18, 0.04]), 'Forzar Gimbal Lock', color='#fee2e2')
        self.btn_diff = Button(plt.axes([0.02, 0.59, 0.18, 0.04]), 'Caso Dif. Máxima', color='#fef3c7')
        self.btn_reset = Button(plt.axes([0.02, 0.54, 0.18, 0.04]), 'Resetear Todo')
        
        self.btn_set_a = Button(plt.axes([0.02, 0.45, 0.08, 0.04]), 'Set Punto A', color='#e0f2fe')
        self.btn_set_b = Button(plt.axes([0.12, 0.45, 0.08, 0.04]), 'Set Punto B', color='#fef2f2')
        self.btn_run_dual = Button(plt.axes([0.02, 0.40, 0.18, 0.04]), 'Comparar Ambos (Dual)', color='#f0fdf4')

        # Presets derecha
        self.btn_p1 = Button(plt.axes([0.85, 0.85, 0.12, 0.04]), 'Vista Cenital')
        self.btn_p2 = Button(plt.axes([0.85, 0.80, 0.12, 0.04]), 'Inclinación 45°')

        # Eventos
        for s in [self.s_roll, self.s_pitch, self.s_yaw]: s.on_changed(self.update)
        self.radio.on_clicked(self.update)
        self.btn_lock.on_clicked(lambda x: self.set_preset(0, 90, 0))
        self.btn_diff.on_clicked(lambda x: self.set_max_diff_case())
        self.btn_reset.on_clicked(lambda x: self.reset())
        self.btn_set_a.on_clicked(lambda x: self.set_point('A'))
        self.btn_set_b.on_clicked(lambda x: self.set_point('B'))
        self.btn_run_dual.on_clicked(lambda x: self.run_interpolation('DUAL'))
        
        self.btn_p1.on_clicked(lambda x: self.set_preset(0, -90, 0))
        self.btn_p2.on_clicked(lambda x: self.set_preset(45, 45, 45))

        self.status_text = self.fig.text(0.02, 0.92, "", fontsize=14, family='monospace', weight='bold')
        self.info_text = self.fig.text(0.02, 0.85, "", fontsize=10, family='monospace', color='#475569')

    def set_preset(self, r, p, y):
        self.s_roll.set_val(r); self.s_pitch.set_val(p); self.s_yaw.set_val(y)

    def set_max_diff_case(self):
        """Caso donde Euler y SLERP son muy distintos: rotación de casi 180 grados."""
        self.set_preset(0, 0, 0)
        self.set_point('A')
        self.set_preset(179, 0, 179)
        self.set_point('B')

    def set_point(self, pt):
        r, p, y = np.radians([self.s_roll.val, self.s_pitch.val, self.s_yaw.val])
        q = RotationEngine.euler_to_quaternion(r, p, y)
        if pt == 'A': self.q_start, self.rpy_start = q, np.array([r, p, y])
        else: self.q_end, self.rpy_end = q, np.array([r, p, y])
        self.update(0)

    def run_interpolation(self, mode_type):
        if self.q_start is None or self.q_end is None: return
        
        for t in np.linspace(0, 1, 60):
            self.ax.cla()
            self.draw_grids()
            self.draw_trajectories()
            
            # Cálculo Euler
            angs = RotationEngine.euler_lerp(self.rpy_start, self.rpy_end, t)
            _, _, M_euler = RotationEngine.get_rotation_matrices(angs[0], angs[1], angs[2])
            
            # Cálculo SLERP
            qt = RotationEngine.slerp(self.q_start, self.q_end, t)
            M_slerp = RotationEngine.quaternion_to_matrix(qt)
            
            # Dibujar ambos modelos
            self.draw_camera(M_euler, base_color='#fb923c', lens_color='#f97316', alpha=0.5)
            self.draw_camera(M_slerp, base_color='#64748b', lens_color='#3b82f6', alpha=1.0)
            self.draw_basis(M_slerp)
            
            self.ax.set_xlim(-1.5, 1.5); self.ax.set_ylim(-1.5, 1.5); self.ax.set_zlim(-1.5, 1.5)
            self.ax.set_axis_off(); self.ax.view_init(elev=20, azim=45)
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def reset(self):
        self.set_preset(0, 0, 0); self.q_start = self.q_end = None; self.update(0)

    def update(self, val):
        r, p, y = np.radians([self.s_roll.val, self.s_pitch.val, self.s_yaw.val])
        mode = self.radio.value_selected
        q_curr = RotationEngine.euler_to_quaternion(r, p, y)
        
        self.ax.cla()
        self.draw_grids()
        self.draw_trajectories()
        
        if mode == 'Método Euler':
            M_yaw, M_pitch, M_roll = RotationEngine.get_rotation_matrices(r, p, y)
            self.draw_ring(self.ring_rads[0], M_yaw, '#06b6d4', plane='XY', alpha=0.2)
            self.draw_ring(self.ring_rads[1], M_pitch, '#22c55e', plane='XZ', alpha=0.4)
            self.draw_ring(self.ring_rads[2], M_roll, '#ec4899', plane='YZ', alpha=0.6)
            self.draw_camera(M_roll); self.draw_basis(M_roll)
            if np.isclose(abs(self.s_pitch.val), 90, atol=0.5):
                self.status_text.set_text("[!] GIMBAL LOCK DETECTADO"); self.status_text.set_color('#ef4444')
            else:
                self.status_text.set_text("MODO: EULER"); self.status_text.set_color('#1e293b')
        else:
            R = RotationEngine.quaternion_to_matrix(q_curr)
            self.draw_camera(R); self.draw_basis(R)
            self.draw_ring(self.ring_rads[1], R, '#a855f7', alpha=0.8)
            self.status_text.set_text("MODO: CUATERNIÓN"); self.status_text.set_color('#3b82f6')

        msg_a = "Set" if self.q_start is not None else "---"
        msg_b = "Set" if self.q_end is not None else "---"
        self.info_text.set_text(f"Punto A: {msg_a} | Punto B: {msg_b}\nq_curr: [{q_curr[0]:.2f}, {q_curr[1]:.2f}, {q_curr[2]:.2f}, {q_curr[3]:.2f}]")
        self.ax.set_xlim(-1.5, 1.5); self.ax.set_ylim(-1.5, 1.5); self.ax.set_zlim(-1.5, 1.5)
        self.ax.set_axis_off(); self.ax.view_init(elev=25, azim=45)
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    app = GimbalApp(); plt.show()