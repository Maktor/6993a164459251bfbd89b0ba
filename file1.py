import numpy as np
import matplotlib.pyplot as plt

def generate_epitrochoid(R, r, d, points=1000):
    theta_max = 10 * np.pi
    t = np.linspace(0, theta_max, points)

    x = (R + r) * np.cos(t) - d * np.cos((R + r) / r * t)
    y = (R + r) * np.sin(t) - d * np.sin((R + r) / r * t)

    return np.column_stack((x, y))

def find_self_intersections(poly_points):
    p1 = poly_points[:-1]
    p2 = poly_points[1:]

    V = p2 - p1

    P_i = p1[:, np.newaxis, :]
    P_j = p1[np.newaxis, :, :]
    V_i = V[:, np.newaxis, :]
    V_j = V[np.newaxis, :, :]

    def cross_2d(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    denom = cross_2d(V_i, V_j)

    safe_denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)

    delta_P = P_j - P_i
    t_num = cross_2d(delta_P, V_j)
    u_num = cross_2d(delta_P, V_i)

    t = t_num / safe_denom
    u = u_num / safe_denom

    n_segs = len(p1)
    i_indices = np.arange(n_segs)[:, np.newaxis]
    j_indices = np.arange(n_segs)[np.newaxis, :]

    mask = (np.abs(denom) > 1e-9) & \
           (t >= 0) & (t <= 1) & \
           (u >= 0) & (u <= 1) & \
           (j_indices > i_indices + 1)

    valid_i = np.where(mask)[0]
    valid_t = t[mask]

    intersections = p1[valid_i] + V[valid_i] * valid_t[:, np.newaxis]
    return intersections

def deduplicate_points(points, tolerance=1e-3):
    if len(points) == 0:
        return points

    unique = []
    for p in points:
        is_new = True
        for existing in unique:
            if np.linalg.norm(p - existing) < tolerance:
                is_new = False
                break
        if is_new:
            unique.append(p)

    return np.array(unique)

pts = generate_epitrochoid(R=3, r=1, d=2, points=800)

raw_crossings = find_self_intersections(pts)
unique_crossings = deduplicate_points(raw_crossings, tolerance=0.1)

plt.figure(figsize=(10, 8))
plt.plot(pts[:, 0], pts[:, 1], c='crimson', lw=1.5)

plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
