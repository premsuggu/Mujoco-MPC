// app/pendulum_mpc.cpp

// Quick test: can we swing up and stabilize a pendulum with MPC + iLQR?

#include "robot_utils.hpp"
#include "ilqr.hpp"
#include "mpc.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

// Make a reference trajectory for the pendulum (either swinging or holding still)
void generateRefs(int num_steps, double dt,
                  std::vector<std::vector<double>>& q_ref,
                  std::vector<std::vector<double>>& v_ref,
                  bool use_sinusoidal = true) {
    q_ref.clear();
    v_ref.clear();
    
    double amplitude = M_PI / 4.0; // 45 degrees swing
    double frequency = 0.5; // 0.5 Hz (slow and smooth)
    
    for (int i = 0; i < num_steps; ++i) {
        double t = i * dt;
        double angle, velocity;
        
        if (use_sinusoidal) {
            angle = amplitude * sin(2.0 * M_PI * frequency * t);
            velocity = amplitude * 2.0 * M_PI * frequency * cos(2.0 * M_PI * frequency * t);
        } else {
            // Hold at 45 degrees (not zero!)
            angle = amplitude;
            velocity = 0.0;
        }
        std::vector<double> q_step = {angle};
        std::vector<double> v_step = {velocity};
        
        q_ref.push_back(q_step);
        v_ref.push_back(v_step);
    }
}

// Save reference trajectories to CSV files
void saveReference(const std::string& q_path, const std::string& v_path,
                   const std::vector<std::vector<double>>& q_ref,
                   const std::vector<std::vector<double>>& v_ref) {
    
    // Save q_ref
    std::ofstream q_file(q_path);
    for (const auto& row : q_ref) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) q_file << ",";
            q_file << row[i];
        }
        q_file << "\n";
    }
    q_file.close();
    
    // Save v_ref (velocity reference)
    std::ofstream v_file(v_path);
    for (const auto& row : v_ref) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) v_file << ",";
            v_file << row[i];
        }
        v_file << "\n";
    }
    v_file.close();
    
    std::cout << "Reference saved to: " << q_path << " and " << v_path << "\n";
}


int main() {
    std::cout << "Let's see if the pendulum can follow a swinging reference using MPC + iLQR...\n";

    // --- Simulation setup ---
    const double dt = 0.02;          // 20Hz MPC
    const int N = 20;                // 1s prediction horizon
    const int sim_steps = 200;       // Simulate for 10 seconds
    const double physics_dt = 0.01;  // Physics engine step size

    // Pendulum parameters (not used directly, but here for clarity)
    const double m = 1.0;            // Mass
    const double L = 1.0;            // Length

    // Load the pendulum model
    RobotUtils robot;
    if (!robot.loadModel("/home/prem/mujoco_mpc/pendulum/pendulum.xml")) {
        std::cerr << "Failed to load pendulum model\n";
        return 1;
    }

    robot.setTimeStep(physics_dt);

    std::cout << "Pendulum model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << "\n";

    // Make and save a swinging reference trajectory
    std::vector<std::vector<double>> q_ref, v_ref;
    generateRefs(sim_steps + N + 10, dt, q_ref, v_ref, true);
    saveReference("/home/prem/mujoco_mpc/pendulum/q_ref.csv",
                  "/home/prem/mujoco_mpc/pendulum/v_ref.csv", q_ref, v_ref);

    // --- Cost weights: tune for your experiment ---
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(robot.nu(), robot.nu());
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());

    Q(0, 0) = 500.0;  // Angle
    Q(1, 1) = 0.1;    // Velocity
    R(0, 0) = 0.01;   // Control effort
    Qf(0, 0) = 10000.0;
    Qf(1, 1) = 10.0;

    robot.setCostWeights(Q, R, Qf);
    std::cout << "Cost weights set for pendulum tracking\n";

    // Load the reference trajectories
    if (!robot.loadReferences("/home/prem/mujoco_mpc/pendulum/q_ref.csv",
                             "/home/prem/mujoco_mpc/pendulum/v_ref.csv")) {
        std::cerr << "Failed to load reference trajectories\n";
        return 1;
    }

    // Set up the MPC controller
    MPC mpc(robot, N, dt);

    // Tweak iLQR solver settings if you want
    auto& ilqr_solver = const_cast<iLQR&>(mpc.solver());
    ilqr_solver.setMaxIterations(1);
    ilqr_solver.setTolerance(1e-4);
    ilqr_solver.setRegularization(1e-6);

    // Log optimal trajectories for later
    mpc.enableOptimalTrajectoryLogging("/home/prem/mujoco_mpc/results");

    std::cout << "Starting pendulum MPC simulation...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Start the pendulum at rest
    Eigen::VectorXd x_current(robot.nx());
    x_current(0) = 0.0;  // Angle
    x_current(1) = 0.0;  // Velocity
    robot.setState(x_current);

    std::cout << "Initial state: angle=" << x_current(0) << " rad ("
              << x_current(0) * 180.0 / M_PI << " deg), velocity=" << x_current(1) << " rad/s\n";

    for (int step = 0; step < sim_steps; ++step) {
        // Get the current state
        robot.getState(x_current);

        // Run one step of MPC
        Eigen::VectorXd u_apply(robot.nu());
        bool success = mpc.stepOnce(x_current, u_apply);

        // Track the cost for this step
        double current_mpc_cost = mpc.getLastSolveCost();

        if (!success) {
            std::cerr << "MPC failed at step " << step << "\n";
            u_apply.setZero();
        }

        // Apply control and step the simulation
        robot.setControl(u_apply);
        robot.step();

        // Print progress every 25 steps (and for the first few)
        if (step % 25 == 0 || step < 5) {
            robot.getState(x_current);
            std::cout << "MPC Step " << step << "/" << sim_steps
                      << " - Cost: " << current_mpc_cost
                      << " - Angle: " << x_current(0) * 180.0 / M_PI << " deg"
                      << " - Control: " << u_apply(0) << " Nm" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Wrap up logging
    mpc.finalizeOptimalTrajectoryLog();

    std::cout << "Pendulum simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / (double)sim_steps << " ms\n";
    std::cout << "Results saved to: /home/prem/mujoco_mpc/results/\n";

    return 0;
}
