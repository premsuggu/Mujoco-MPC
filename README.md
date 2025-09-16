# Mujoco-MPC

- **C++ MPC and iLQR implementation** for both humanoid and pendulum robots
- **Reference tracking** and cost tuning, with easy CSV logging
- **Python scripts** for plotting and visualizing your robot's performance
- **Ready-to-run examples** for both humanoid and pendulum

## How to Use

1. **Build the C++ code** (requires MuJoCo, Eigen, CMake):
   ```bash
   mkdir -p build && cd build
   cmake ..
   make
   ```
2. **Run a simulation** (from the build folder):
   - Humanoid: `./humanoid_mpc`
   - Pendulum: `./pendulum_mpc`

3. **Visualize results**:
   - Use the Python scripts in the main folder to plot or animate the results. For example:
     ```bash
     python3 humanoid_plotter.py
     python3 simulate.py
     ```

## Project Structure

- `app/` — Main C++ entry points for each robot
- `src/` — Core MPC, iLQR, and robot utilities
- `include/` — C++ headers
- `data/` — Reference trajectories
- `results/` — Output logs and plots
- `robot/` and `pendulum/` — Model files

## Tips
- Tweak cost weights in the C++ files to see how tracking changes
- All logs and plots go in the `results/` folder
- The code is meant to be readable and hackable—experiment away!

## Requirements
- MuJoCo
- Eigen
- CMake
- Python 3 (for plotting/visualization)
- Python packages: `numpy`, `pandas`, `matplotlib`, `mujoco`

---