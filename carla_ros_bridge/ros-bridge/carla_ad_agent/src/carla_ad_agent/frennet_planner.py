"""

Modified Frenet optimal trajectory generator to work with ROS and Carla

author: Mukul Chodhary

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import copy
import math

from Polynomials import QuinticPolynomial, QuarticPolynomial
from CubicSpline import cubic_spline_planner
from misc import distance_vehicle  # pylint: disable=relative-import
# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0



            
class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

class FrenetPlanner:
    def __init__(self, ego_bbox_extents, 
                 MaxCurvature=1, D_ROAD_W = 1.0, 
                 MaxAccel=6.0, DT=0.2, MAX_T=2, MaxSpeed=100.0,
                 MIN_T=0.5, D_T_S=4.0/3.6, N_S_SAMPLE=1):
        self.ego_bbox_extents = ego_bbox_extents
        # params
        self.MAX_ACCEL = MaxAccel  # maximum acceleration [m/ss]
        self.MAX_CURVATURE = MaxCurvature  # maximum curvature [1/m]
        self.MAX_ROAD_WIDTH = None # maximum road width [m]
        self.D_ROAD_W = D_ROAD_W  # road width sampling length [m]
        self.DT = DT  # time tick [s]
        self.MAX_T = MAX_T  # max prediction time [m]
        self.MIN_T = MIN_T  # min prediction time [m]
        self.MAX_SPEED = MaxSpeed # maximum speed [m/s]
        
        self.D_T_S = D_T_S  # target speed sampling length [m/s]
        self.N_S_SAMPLE = N_S_SAMPLE  # sampling number of target speedk
        
        self.c_d = 0.0 # current lateral position [m]
        self.c_d_d = 0.0 # current lateral speed [m/s]
        self.c_d_dd = 0.0 # current lateral acceleration [m/s]
        self.c_speed = 1 # current speed [m/s]
        self.c_accel = 0.0 # current acceleration [m/ss]
        self.csp  = None # current course 
        self.s0 = 0.0 # current course position

       
            
    def calc_frenet_paths(self, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
        frenet_paths = []

        # generate path to each offset goal
        for di in np.arange(-self.MAX_ROAD_WIDTH, self.MAX_ROAD_WIDTH, self.D_ROAD_W):

            # Lateral motion planning
            for Ti in np.arange(self.MIN_T, self.MAX_T, self.DT):
                fp = FrenetPath()

                # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, self.DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(self.TARGET_SPEED - self.D_T_S * self.N_S_SAMPLE,
                                    self.TARGET_SPEED + self.D_T_S * self.N_S_SAMPLE, self.D_T_S):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (self.TARGET_SPEED - tfp.s_d[-1]) ** 2

                    tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                    tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                    tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths


    def calc_global_paths(self, fplist, csp):
        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                # if not self._prune_point(fp, fx, fy):
                #     continue
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))

            if len(fp.x) <= 1 or len(fp.yaw) == 0:
                continue
            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist

    def check_paths(self, fplist, ob):
        ok_ind = []
        for i, _ in enumerate(fplist):
           
            if any([v > self.MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                continue
            elif any([abs(a) > self.MAX_ACCEL for a in
                    fplist[i].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > self.MAX_CURVATURE for c in
                    fplist[i].c]):  # Max curvature check
                continue
            # print("\x1b[1;31m" + "Frenet path check passed, now check collision" + "\x1b[0m")
            # # collision check
            collisions = False
            for obstacle in ob:
                if obstacle.check_collision(fp = fplist[i], 
                                            ego_bbox_extents = self.ego_bbox_extents,
                                            dt = self.DT, ego_v = self.c_speed, lane_width = self.MAX_ROAD_WIDTH):
                    collisions = True 
                    break
            if collisions:  
                print("collision ")
                continue

            ok_ind.append(i)

        return [fplist[i] for i in ok_ind]
    
    def prune_paths(self, fplist):
        sampling_radius = self.c_speed * 1 / 3.6  # 0.5 seconds horizon
        min_distance = sampling_radius * 0.8
        # remove all points in a path that are too close to each other
        for i, _ in enumerate(fplist):
            if len(fplist[i].x) <= 1:
                continue
            j = 0
            lenth = len(fplist[i].x) -1
            while j < lenth:
                if math.hypot(fplist[i].x[j+1] - fplist[i].x[j], fplist[i].y[j+1] - fplist[i].y[j]) < min_distance:
                    fplist[i].x.pop(j+1)
                    fplist[i].y.pop(j+1)
                    fplist[i].yaw.pop(j+1)
                    fplist[i].ds.pop(j+1)
                    fplist[i].c.pop(j)
                    lenth -= 1
                else:
                    j += 1
        return fplist
    
    def _prune_point(self, fp, x, y):
        sampling_radius = self.c_speed * 0.5 / 3.6  # 0.5 seconds horizon
        min_distance = sampling_radius * 0.9
        # is this point too close to any other point in the path?
        for i in range(len(fp.x)):
            if math.hypot(fp.x[i] - x, fp.y[i] - y) < sampling_radius:
                return True
        return False
       


    def frenet_optimal_planning(self, c_pos, c_speed, c_accel, ob, target_speed, road_width):
        """
        frenet_optimal_planning
        input
            c_speed: current speed [m/s]
            ob: obstacle list
            target_speed: target speed [m/s]
            road_width: road width [m]
            s0: current_s
            output
            path_exists: bool
            x: next_x position
            y: next_y position
        """
        if target_speed == None:
            self.TARGET_SPEED = 0
        self.TARGET_SPEED = target_speed
       
        self.MAX_ROAD_WIDTH = road_width
        prev_c_d = self.c_d
        self.s0, _ = self.csp.calc_sd_from_xy(c_pos.x, c_pos.y, self.s0)
        print("\x1b[1;31m" + "Previous lateral position: " + str(prev_c_d) + "\x1b[0m")
        print("\x1b[1;31m" + "Current lateral position: " + str(self.c_d) + "\x1b[0m")
        print(self.s0, self.c_d)
        
        self.c_accel = c_accel
        self.c_speed = c_speed
        # print("\x1b[1;31m" + "Current speed: " + str(self.c_speed) + "\x1b[0m")
        # print("\x1b[1;31m" + "Current acceleration: " + str(self.c_accel) + "\x1b[0m")
        # print("\x1b[1;31m" + "Current lateral position: " + str(self.c_d) + "\x1b[0m")
        # print("\x1b[1;31m" + "Current lateral speed: " + str(self.c_d_d) + "\x1b[0m")
        # print("\x1b[1;31m" + "Current lateral acceleration: " + str(self.c_d_dd) + "\x1b[0m")
        # print("\x1b[1;31m" + "Current course position: " + str(self.s0) + "\x1b[0m")
        # print("\x1b[1;31m" + "Current target speed: " + str(self.TARGET_SPEED) + "\x1b[0m")
        # print("\x1b[1;31m" + "Current road width: " + str(self.MAX_ROAD_WIDTH) + "\x1b[0m")
        fplist = self.calc_frenet_paths(self.c_speed, self.c_accel, s0=self.s0,
                                        c_d = self.c_d, c_d_d=self.c_d_d, c_d_dd = self.c_d_dd)
        if len(fplist) == 0 or len(fplist[0].s) == 0:
            return False, None, None
        print("\x1b[1;31m" + "Frenet paths generated" + "\x1b[0m")
        print("first: ", len(fplist[0].s))
        fplist = self.calc_global_paths(fplist, self.csp)
        print("second: ", len(fplist[0].s))
        print(fplist[0].x, fplist[0].y)
        fplist = self.check_paths(fplist, ob)
        print("third: ", len(fplist))

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        path_exists = False
        x,y = None, None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp
                path_exists = True
        if path_exists and len(best_path.x)>=1:
            x,y = self.set_from_best_path(best_path)

        return path_exists, x,y, best_path
    
    def set_from_best_path(self, best_path):
        i = 1
        if(len(best_path.x) == 1):
            i = 0
        x,y = best_path.x[i], best_path.y[i]
        self.s0 = best_path.s[i]
        self.c_d = best_path.d[i]
        self.c_d_d = best_path.d_d[i]
        self.c_d_dd = best_path.d_dd[i]
        return x,y

    def generate_target_course(self, waypoint_buffer):
          # target waypoints
        x = []
        y = []
        for i in waypoint_buffer:
            # dont append waypoints that are too close to waypoints already in buffer
            if len(x) > 0:
                if np.hypot(i.position.x - x[-1], i.position.y - y[-1]) <= 0.1:
                    continue

            x.append(i.position.x)
            y.append(i.position.y)
        # print('\x1b[1;31m' + "Waypoints" + '\x1b[0m')
        # print(x)
        # print(y)
        csp = cubic_spline_planner.CubicSpline2D(x, y)
        self.csp = csp
        self.s0 = 0.0
        s = np.arange(0, csp.s[-1], 0.1)

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))

        return s, rx, ry, ryaw, rk, csp


# def main():
#     print(__file__ + " start!!")

#     # way points
#     wx = [0.0, 10.0, 20.5, 35.0, 70.5]
#     wy = [0.0, -6.0, 5.0, 6.5, 0.0]
#     # obstacle lists
#     ob = np.array([[20.0, 10.0],
#                    [30.0, 6.0],
#                    [30.0, 8.0],
#                    [35.0, 8.0],
#                    [50.0, 3.0]
#                    ])

#     tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

#     # initial state
#     c_speed = 10.0 / 3.6  # current speed [m/s]
#     c_accel = 0.0  # current acceleration [m/ss]
#     c_d = 2.0  # current lateral position [m]
#     c_d_d = 0.0  # current lateral speed [m/s]
#     c_d_dd = 0.0  # current lateral acceleration [m/s]
#     s0 = 0.0  # current course position

#     area = 20.0  # animation area length [m]

#     for i in range(SIM_LOOP):
#         path = frenet_optimal_planning(
#             csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob)

#         s0 = path.s[1]
#         c_d = path.d[1]
#         c_d_d = path.d_d[1]
#         c_d_dd = path.d_dd[1]
#         c_speed = path.s_d[1]
#         c_accel = path.s_dd[1]

#         if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
#             print("Goal")
#             break

#         if show_animation:  # pragma: no cover
#             plt.cla()
#             # for stopping simulation with the esc key.
#             plt.gcf().canvas.mpl_connect(
#                 'key_release_event',
#                 lambda event: [exit(0) if event.key == 'escape' else None])
#             plt.plot(tx, ty)
#             plt.plot(ob[:, 0], ob[:, 1], "xk")
#             plt.plot(path.x[1:], path.y[1:], "-or")
#             plt.plot(path.x[1], path.y[1], "vc")
#             plt.xlim(path.x[1] - area, path.x[1] + area)
#             plt.ylim(path.y[1] - area, path.y[1] + area)
#             plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
#             plt.grid(True)
#             plt.pause(0.0001)

#     print("Finish")
#     if show_animation:  # pragma: no cover
#         plt.grid(True)
#         plt.pause(0.0001)
#         plt.show()



