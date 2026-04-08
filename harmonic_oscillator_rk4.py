import numpy as np
import matplotlib.pyplot as plt

# ——————————————————————————————————————————————————————————————––––
# Simple implementation of RK4 to solve 1D harmonic oscillator:    
#
#   x'' + w^2 x = 0  ->  [x', y'] = [v, -w^2 x]
# 
# ——————————————————————————————————————————————————————————————————

# Takes y = [v, x] and gives d[x,y]/dt = [x',y']
def ho_deriv(t, y, omega):

    x, v = y;

    return np.array([v, -omega**2 * x])

# Computes the next step according to RK4 algorithm
def rk4_step(f, t, dt, y, **params):

    k1 = f(t, y, **params);
    k2 = f(t + dt/2, y + dt/2 * k1, **params);
    k3 = f(t + dt/2, y + dt/2 * k2, **params);
    k4 = f(t + dt, y + dt * k3, **params);

    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Computes the solution array
def solve(f, y0, t_span, dt, **params):

    t_vec = np.arange(t_span[0], t_span[1], dt);

    y = np.zeros((len(t_vec), len(y0)));
    y[0] = y0;

    for i in range(1, len(t_vec)):
        y[i] = rk4_step(f, t_vec[i-1], dt, y[i-1], **params);

    return t_vec, y

if __name__ == "__main__":

    omega = 1.0;
    dt = 0.01;
    t_final = 2*np.pi;
    y0 = np.array([1.0,0.0]);

    t_vec, y = solve(ho_deriv, y0, (0, t_final), dt, omega=omega)
    
    num_sol = y[:, 0];
    anal_sol = np.cos(omega * t_vec);

    plt.figure()
    plt.plot(t_vec, num_sol, label='RK4')
    plt.plot(t_vec, anal_sol, '--', label='Analytical')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Harmonic Oscillator')
    plt.legend()
    plt.show()

