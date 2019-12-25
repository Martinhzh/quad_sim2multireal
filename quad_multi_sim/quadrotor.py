#!/usr/bin/env python
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
Also see: D. Mellinger, N. Michael, V.Kumar. 
Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors
http://journals.sagepub.com/doi/pdf/10.1177/0278364911434236

Developers:
James Preiss, Artem Molchanov, Tao Chen 

References:
[1] RotorS: https://www.researchgate.net/profile/Fadri_Furrer/publication/309291237_RotorS_-_A_Modular_Gazebo_MAV_Simulator_Framework/links/5a0169c4a6fdcc82a3183f8f/RotorS-A-Modular-Gazebo-MAV-Simulator-Framework.pdf
[2] CrazyFlie modelling: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
[3] HummingBird: http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-hummingbird/
[4] CrazyFlie thrusters transition functions: https://www.bitcraze.io/2015/02/measuring-propeller-rpm-part-3/
[5] HummingBird modelling: https://digitalrepository.unm.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1189&context=ece_etds
[6] Rotation b/w matrices: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
[7] Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
"""
import argparse
import logging
import sys
import time

import matplotlib.pyplot as plt
import transforms3d as t3d
from numpy.linalg import norm

from quad_multi_sim.multi_quad import MultiQuadEnv, MultiQuadPolicy
from quad_sim.inertia import QuadLink
from quad_sim.quad_utils import *
from quad_sim.quadrotor_control import *


logger = logging.getLogger(__name__)


class DummyPolicy(object):
    def __init__(self, dt=0.01, switch_time=2.5):
        self.action = np.zeros([4, ])
        self.dt = 0.

    def step(self, x):
        return self.action

    def reset(self):
        pass

class UpDownPolicy(object):
    def __init__(self, dt=0.01, switch_time=2.5):
        self.t = 0
        self.dt=dt
        self.switch_time = switch_time
        self.action_up =  np.ones([4,])
        self.action_up[:2] = 0.
        self.action_down =  np.zeros([4,])
        self.action_down[:2] = 1.
    
    def step(self, x):
        self.t += self.dt
        if self.t < self.switch_time:
            return self.action_up
        else:
            return self.action_down
    def reset(self):
        self.t = 0.

def test_rollout(num_quads, quad, dyn_randomize_every=None, dyn_randomization_ratio=None,
    render=True, traj_num=10, plot_step=None, plot_dyn_change=True, plot_thrusts=False,
    sense_noise=None, policy_type="mellinger", init_random_state=False, obs_repr="xyz_vxyz_rot_omega",csv_filename=None):
    import tqdm
    #############################
    # Init plottting
    if plot_step is not None:
        fig = plt.figure(1)
        # ax = plt.subplot(111)
        plt.show(block=False)

    # render = True
    # plot_step = 50
    time_limit = 25
    render_each = 2
    rollouts_num = traj_num
    plot_obs = False

    if policy_type == "mellinger":
        raw_control=False
        raw_control_zero_middle=True
        policy = MultiQuadPolicy(DummyPolicy, num_quads) #since internal Mellinger takes care of the policy
    elif policy_type == "updown":
        raw_control=True
        raw_control_zero_middle=False
        policy = MultiQuadPolicy(UpDownPolicy, num_quads)
    else:
        raise Exception('Policy type unknown')

    # zhehui: Initial Environment
    # Create multi Env --- array
    # Create multi Env --- array
    env = MultiQuadEnv(
        num_quads, quad, raw_control, raw_control_zero_middle,
        dyn_randomize_every=dyn_randomize_every,
        dyn_randomization_ratio=dyn_randomization_ratio,
        sense_noise=sense_noise, init_random_state=init_random_state, obs_repr=obs_repr,
    )

    # zhehui: dt = time * radius frequency
    policy.dt = 1./ env.control_freq

    env.max_episode_steps = time_limit
    print('Reseting env ...')

    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        # print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        # print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)
        # TODO: print obs/action space of the first quadrotor in multi-quad env?
        pass
    # input('Press any key to continue ...')

    ## Collected statistics for dynamics
    dyn_param_names = [
        "mass",
        "inertia",
        "thrust_to_weight",
        "torque_to_thrust",
        "thrust_noise_ratio",
        "vel_damp",
        "damp_omega_quadratic"
    ]

    # zhehui: Create list based on num of quadrotors
    dyn_param_stats = [[] for i in dyn_param_names]

    action = [np.array([0.0, 0.5, 0.0, 0.5])] * num_quads
    rollouts_id = 0

    start_time = time.time()
    # while rollouts_id < rollouts_num:
    for rollouts_id in tqdm.tqdm(range(rollouts_num)):
        rollouts_id += 1
        s = env.reset()
        policy.reset()
        ## Diagnostics
        observations = []
        velocities = []
        actions = []

        ## Collecting dynamics params
        if plot_dyn_change:
            for par_i, par in enumerate(dyn_param_names):
                dyn_param_stats[par_i].append(np.array(getattr(env.dynamics, par)).flatten())
                # print(par, dyn_param_stats[par_i][-1])

        t = 0
        while True:
            if render and (t % render_each == 0):
                print('Render envs at timestep:', t)
                env.render()
            action = policy.step(s)
            s, r, done, info = env.step(action)
            
            actions.append(action)
            observations.append(s)
            # print('Step: ', t, ' Obs:', s)
            # quat = R2quat(rot=s[6:15])
            # csv_data.append(np.concatenate([np.array([1.0/env.control_freq * t]), s[0:3], quat]))

            if plot_step is not None and t % plot_step == 0:
                plt.clf()

                if plot_obs:
                    observations_arr = np.array(observations)
                    # print('observations array shape', observations_arr.shape)
                    dimenstions = observations_arr.shape[1]
                    for dim in range(dimenstions):
                        plt.plot(observations_arr[:, dim])
                    plt.legend([str(x) for x in range(observations_arr.shape[1])])

                plt.pause(0.05) #have to pause otherwise does not draw
                plt.draw()

            if all(done):
                break

            t += 1

        # if plot_thrusts:
        #     plt.figure(3, figsize=(10, 10))
        #     ep_time = np.linspace(0, policy.dt * len(actions), len(actions))
        #     actions = np.array(actions)
        #     thrusts = np.array(thrusts)
        #     for i in range(4):
        #         plt.plot(ep_time, actions[:,i], label="Thrust desired %d" % i)
        #         plt.plot(ep_time, thrusts[:,i], label="Thrust produced %d" % i)
        #     plt.legend()
        #     plt.show(block=False)
        #     input("Press Enter to continue...")
        
        # if csv_filename is not None:
        #     import csv
        #     with open(csv_filename, mode="w") as csv_file:
        #         csv_writer = csv.writer(csv_file, delimiter=',')
        #         for row in csv_data:
        #             csv_writer.writerow([i for i in row])


    if plot_dyn_change:
        dyn_par_normvar = []
        dyn_par_means = []
        dyn_par_var = []
        plt.figure(2, figsize=(10, 10))
        for par_i, par in enumerate(dyn_param_stats):
            plt.subplot(3, 3, par_i+1)
            par = np.array(par)

            ## Compute stats
            # print(dyn_param_names[par_i], par)
            dyn_par_means.append(np.mean(par, axis=0))
            dyn_par_var.append(np.std(par, axis=0))
            dyn_par_normvar.append(dyn_par_var[-1] / dyn_par_means[-1])

            if par.shape[1] > 1:
                for vi in range(par.shape[1]):
                    plt.plot(par[:, vi])
            else:
                plt.plot(par)
            # plt.title(dyn_param_names[par_i] + "\n Normvar: %s" % str(dyn_par_normvar[-1]))
            plt.title(dyn_param_names[par_i])
            print(dyn_param_names[par_i], "NormVar: ", dyn_par_normvar[-1])
    
    print("##############################################################")
    print("Total time: ", time.time() - start_time )

    # print('Rollouts are done!')
    # plt.pause(2.0)
    # plt.waitforbuttonpress()
    if plot_step is not None or plot_dyn_change:
        plt.show(block=False)
        input("Press Enter to continue...")

def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-n', '--num',
        type=int,
        default=2,
        help='Number of drones to simulate!',
    )
    parser.add_argument(
        '-m',"--mode",
        default="mellinger",
        help="Test mode: "
             "mellinger - rollout with mellinger controller"
             "updown - rollout with UpDown controller (to test step responses)"
    )
    parser.add_argument(
        '-q',"--quad",
        default="defaultquad",
        help="Quadrotor model to use: \n" + 
            "- defaultquad \n" + 
            "- crazyflie \n" +
            "- random"
    )
    parser.add_argument(
        '-dre',"--dyn_randomize_every",
        type=int,
        help="How often (in terms of trajectories) to perform randomization"
    )
    parser.add_argument(
        '-drr',"--dyn_randomization_ratio",
        type=float,
        default=0.5,
        help="Randomization ratio for random sampling of dynamics parameters"
    )
    parser.add_argument(
        '-r',"--render",
        action="store_false",
        help="Use this flag to turn off rendering"
    )
    parser.add_argument(
        '-trj',"--traj_num",
        type=int,
        default=10,
        help="Number of trajectories to run"
    )
    parser.add_argument(
        '-plt',"--plot_step",
        type=int,
        help="Plot step"
    )
    parser.add_argument(
        '-pltdyn',"--plot_dyn_change",
        action="store_true",
        help="Plot the dynamics change from trajectory to trajectory?"
    )
    parser.add_argument(
        '-pltact',"--plot_actions",
        action="store_true",
        help="Plot actions commanded and thrusts produced after damping"
    )
    parser.add_argument(
        '-sn',"--sense_noise",
        action="store_false",
        help="Add sensor noise? Use this flag to turn the noise off"
    )
    parser.add_argument(
        '-irs',"--init_random_state",
        action="store_true",
        help="Add sensor noise?"
    )
    parser.add_argument(
        '-csv',"--csv_filename",
        help="Filename for qudrotor data"
    )
    parser.add_argument(
        '-o',"--obs_repr",
        default="xyz_vxyz_rot_omega_acc_act",
        help="State components. Options:\n" +
             "xyz_vxyz_rot_omega" +
             "xyz_vxyz_rot_omega_act" +
             "xyz_vxyz_rot_omega_acc_act" 
    )
    args = parser.parse_args()

    if args.sense_noise:
        sense_noise="default"
    else:
        sense_noise=None

    print('Running test rollout ...')
    test_rollout(
        num_quads=args.num,
        quad=args.quad, 
        dyn_randomize_every=args.dyn_randomize_every,
        dyn_randomization_ratio=args.dyn_randomization_ratio,
        render=args.render,
        traj_num=args.traj_num,
        plot_step=args.plot_step,
        plot_dyn_change=args.plot_dyn_change,
        plot_thrusts=args.plot_actions,
        sense_noise=sense_noise,
        policy_type=args.mode,
        init_random_state=args.init_random_state,
        obs_repr=args.obs_repr,
        csv_filename=args.csv_filename,
    )


if __name__ == '__main__':
    main(sys.argv)
