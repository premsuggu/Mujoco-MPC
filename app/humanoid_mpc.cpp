// app/humanoid_mpc.cpp
#include "robot_utils.hpp"
#include "ilqr.hpp"
#include "mpc.hpp"
#include <iostream>
#include <fstream>
#include <chrono>


int main() {
    // --- Simulation setup ---
    const double dt = 0.02;          // MPC runs at 50Hz
    const int N = 25;                // 0.5s prediction horizon
    const int sim_steps = 50;        // Simulate for 1 second
    const double physics_dt = 0.02;  // Physics engine step size

    // Fire up the robot model
    RobotUtils robot;
    if (!robot.loadModel("/home/prem/mujoco_mpc/robot/h1_description/mjcf/scene.xml")) {
        std::cerr << "Failed to load robot model\n";
        return 1;
    }

    // Tweak simulation parameters for stability and match the paper
    robot.setContactImpratio(100.0);
    robot.setTimeStep(physics_dt);
    robot.setGravity(0.0, 0.0, -9.81);            // Small gravity to stabilize the motion
    // robot.scaleRobotMass(0.3); // Uncomment to make the robot featherweight

    // Start in a nice, stable standing pose
    robot.initializeStandingPose();

    std::cout << "Model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << "\n";

    // --- Cost weights: tune these for your needs ---
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(robot.nu(), robot.nu());
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());

    // Position weights
    Q(0,0) = 10.0;   // X
    Q(1,1) = 10.0;   // Y
    Q(2,2) = 50.0;   // Z

    // Orientation weights
    Q(3,3) = 100.0;  // qw
    Q(4,4) = 100.0;  // qx
    Q(5,5) = 100.0;  // qy
    Q(6,6) = 50.0;   // qz

    // Joint positions
    for (int i = 7; i < robot.nq(); ++i) {
        Q(i, i) = 50.0;
    }

    // Velocity weights (important for floating robots)
    int nq = robot.nq();
    Q(nq + 0, nq + 0) = 100.0;  // X velocity
    Q(nq + 1, nq + 1) = 100.0;  // Y velocity
    Q(nq + 2, nq + 2) = 300.0;  // Z velocity
    Q(nq + 3, nq + 3) = 50.0;   // Angular velocities
    Q(nq + 4, nq + 4) = 50.0;
    Q(nq + 5, nq + 5) = 50.0;

    // Joint velocities
    for (int i = nq + 6; i < robot.nx(); ++i) {
        Q(i, i) = 100.0;
    }

    R *= 0.1;         // Penalize control effort a bit more
    Qf = Q*10.0;      // Terminal cost is higher
    Qf(nq + 2, nq + 2)  *= 10.0;

    robot.setCostWeights(Q, R, Qf);

    // No joint or control limit penalties for this run
    robot.setConstraintWeights(0, 0);

    // Load reference trajectories (standing still)
    if (!robot.loadReferences("/home/prem/mujoco_mpc/data/q_standing.csv", "/home/prem/mujoco_mpc/data/v_standing.csv")) {
        std::cerr << "Failed to load reference trajectories\n";
        return 1;
    }

    // Set up the MPC controller
    MPC mpc(robot, N, dt);

    // Log optimal trajectories for later analysis
    mpc.enableOptimalTrajectoryLogging("/home/prem/mujoco_mpc/results");

    auto start_time = std::chrono::high_resolution_clock::now();

    int physics_steps_per_mpc = (int)(dt / physics_dt);
    for (int step = 0; step < sim_steps; ++step) {
        // Grab the current state
        Eigen::VectorXd x_current(robot.nx());
        robot.getState(x_current);

        // If the state goes off the rails, bail out
        if (!x_current.allFinite()) {
            std::cerr << "NaN detected in state at step " << step << ", breaking simulation" << std::endl;
            break;
        }

        // Run one step of MPC
        Eigen::VectorXd u_apply(robot.nu());
        bool success = mpc.stepOnce(x_current, u_apply);

        if (!success) {
            std::cerr << "MPC failed at step " << step << "\n";
            // If it fails, just use zero control for a bit
            u_apply.setZero();
            if (step > 10) {
                break;
            }
        }

        // If control is NaN, zero it out
        if (!u_apply.allFinite()) {
            std::cerr << "NaN detected in control at step " << step << ", using zero control" << std::endl;
            u_apply.setZero();
        }

        robot.setControl(u_apply);
        // Recompute contacts after changing control
        mj_forward(robot.model(), robot.data());
        for (int sub_step = 0; sub_step < physics_steps_per_mpc; ++sub_step) {
            robot.step();
        }

        // Print progress every step
        if (step % 1 == 0 || step == sim_steps - 1) {
            double current_cost = mpc.getLastSolveCost();
            double z_position = x_current(2);
            double x_position = x_current(0);
            double y_position = x_current(1);
            double u_min = u_apply.minCoeff();
            double u_max = u_apply.maxCoeff();

            std::cout << "Step " << step << "/" << sim_steps
                      << " | Cost: " << current_cost
                      << " | (X,Y,Z): (" << x_position << "," << y_position << "," << z_position << ") m"
                      << " | Control range: [" << u_min << ", " << u_max << "]"
                      << std::endl;
            // robot.diagnoseContactForces();
            robot.debugContactSolver();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Wrap up logging
    mpc.finalizeOptimalTrajectoryLog();

    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / (double)sim_steps << " ms\n";

    return 0;
}
