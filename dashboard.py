import numpy as np
import matplotlib.pyplot as plt


class MissileSim:
    def __init__(self, dt=0.01, N=4.0, missile_speed=800.0):
        self.dt = dt
        self.N = N
        self.missile_speed = missile_speed

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else np.zeros_like(v)

    def run_simulation(self, m_pos, t_pos, t_vel, max_time=30.0):
        m_pos = np.array(m_pos, dtype=float)
        t_pos = np.array(t_pos, dtype=float)
        t_vel = np.array(t_vel, dtype=float)

        
        m_vel = self._normalize(t_pos - m_pos) * self.missile_speed
        r_prev = t_pos - m_pos

        history = []

        for t in np.arange(0, max_time, self.dt):
            r = t_pos - m_pos
            dist = np.linalg.norm(r)

            if dist < 5.0:
                print(f"Target Intercepted at t = {t:.2f} seconds!")
                break

            v_rel = t_vel - m_vel
            r_norm = self._normalize(r)
            closing_speed = -np.dot(v_rel, r_norm)

            los_rate = (r - r_prev) / self.dt
            r_prev = r.copy()

            acc = los_rate * (self.N * closing_speed)

            
            m_vel = self._normalize(m_vel + acc * self.dt) * self.missile_speed
            m_pos = m_pos + m_vel * self.dt
            t_pos = t_pos + t_vel * self.dt

            history.append(np.concatenate(([t], m_pos, t_pos)))

        return np.array(history)


def main():
    
    m_initial_pos = [0.0, 0.0, 0.0]
    t_initial_pos = [5000.0, 4000.0, 2000.0]
    t_initial_vel = [-150.0, -50.0, 0.0]

    
    sim = MissileSim()
    sim_data = sim.run_simulation(m_initial_pos, t_initial_pos, t_initial_vel)

    if len(sim_data) == 0:
        print("Simulation returned no data points.")
        return

    time_stamps = sim_data[:, 0]
    missile_coords = sim_data[:, 1:4]
    target_coords = sim_data[:, 4:7]

    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("3D Missile Trajectory & Interception Simulator (Pure Python)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")

    ax.plot(
        missile_coords[:, 0],
        missile_coords[:, 1],
        missile_coords[:, 2],
        label="Missile Path",
        color="red",
        linewidth=2,
    )
    ax.plot(
        target_coords[:, 0],
        target_coords[:, 1],
        target_coords[:, 2],
        label="Target Path",
        color="blue",
        linestyle="--",
    )
    ax.scatter(
        missile_coords[-1, 0],
        missile_coords[-1, 1],
        missile_coords[-1, 2],
        color="green",
        s=100,
        label="Intercept Point",
    )

    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()