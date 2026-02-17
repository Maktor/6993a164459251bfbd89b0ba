import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

R = 3
r = 1
d = 2

def x_eq(t):
    return (R + r) * np.cos(t) - d * np.cos((R + r) / r * t)

def y_eq(t):
    return (R + r) * np.sin(t) - d * np.sin((R + r) / r * t)

def dx_dt(t):
    return -(R + r) * np.sin(t) + d * ((R + r) / r) * np.sin((R + r) / r * t)

def dy_dt(t):
    return (R + r) * np.cos(t) - d * ((R + r) / r) * np.cos((R + r) / r * t)

def find_distinct_roots(func, t_min, t_max, num_samples=1000, tol=1e-5):
    """
    Finds roots of func(t) = 0.
    Scans the domain for sign changes and uses brentq for precision.
    """
    t_vals = np.linspace(t_min, t_max, num_samples)
    roots = []
    
    for i in range(len(t_vals) - 1):
        t1, t2 = t_vals[i], t_vals[i+1]
        y1, y2 = func(t1), func(t2)
        
        if y1 * y2 <= 0:
            try:
                root = optimize.brentq(func, t1, t2)
                if not roots or abs(root - roots[-1]) > tol:
                    roots.append(root)
            except ValueError:
                pass
                
    return np.array(roots)

t_roots_for_A = find_distinct_roots(y_eq, 0, 2 * np.pi)

points_A = []
for t in t_roots_for_A:
    pt = (round(x_eq(t), 4), round(y_eq(t), 4)) 
    points_A.append(pt)

unique_A = list(set(points_A))
A = len(unique_A)

t_roots_for_B = find_distinct_roots(x_eq, 0, 2 * np.pi)

points_B = []
for t in t_roots_for_B:
    pt = (round(x_eq(t), 4), round(y_eq(t), 4))
    points_B.append(pt)

unique_B = list(set(points_B))
B = len(unique_B)

def find_self_intersections_robust(num_points=1200):
    t = np.linspace(0, 2*np.pi, num_points)
    x = x_eq(t)
    y = y_eq(t)
    points = np.column_stack((x, y))
    
    intersections = []
    
    for i in range(len(points) - 2):
        for j in range(i + 2, len(points) - 1):
            if j == i + 1: continue
            
            p1, p2 = points[i], points[i+1]
            p3, p4 = points[j], points[j+1]
            
            det = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
            
            if abs(det) < 1e-9: continue # Parallel lines
            
            denom = det
            t_ua = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / denom
            t_ub = ((p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])) / denom
            
            if 0 < t_ua < 1 and 0 < t_ub < 1:
                ix = p1[0] + t_ua * (p2[0] - p1[0])
                iy = p1[1] + t_ua * (p2[1] - p1[1])
                intersections.append((round(ix, 3), round(iy, 3)))

    return list(set(intersections))

unique_C = find_self_intersections_robust()
C = len(unique_C)

total = A + B + C

print(f"-"*30)
print(f"A (X-axis intercepts): {A} -> Points: {unique_A}")
print(f"B (Y-axis intercepts): {B} -> Points: {unique_B}")
print(f"C (Self-intersections): {C} -> Points: {unique_C}")
print(f"-"*30)
print(f"FINAL ANSWER (A+B+C): {total}")
print(f"-"*30)

t_plot = np.linspace(0, 2*np.pi, 1000)
plt.figure(figsize=(6,6))
plt.plot(x_eq(t_plot), y_eq(t_plot), label='Epitrochoid')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)

xA, yA = zip(*unique_A) if unique_A else ([],[])
xB, yB = zip(*unique_B) if unique_B else ([],[])
xC, yC = zip(*unique_C) if unique_C else ([],[])

plt.scatter(xA, yA, color='red', s=50, zorder=5, label=f'A (Count: {A})')
plt.scatter(xB, yB, color='blue', s=50, zorder=5, label=f'B (Count: {B})')
plt.scatter(xC, yC, color='green', s=50, zorder=5, label=f'C (Count: {C})')

plt.title(f"Visual Verification: Total = {total}")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
