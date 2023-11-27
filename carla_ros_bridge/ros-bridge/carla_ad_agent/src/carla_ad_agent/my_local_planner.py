#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
This module contains a local planner to perform
low-level waypoint following based on PID controllers.
"""

from collections import deque
import rospy
import math
import numpy as np
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from carla_waypoint_types.srv import GetWaypoint
from carla_msgs.msg import CarlaEgoVehicleControl
from vehicle_pid_controller import VehiclePIDController  # pylint: disable=relative-import
from misc import distance_vehicle  # pylint: disable=relative-import
import carla
import carla_ros_bridge.transforms as trans
import shapely.geometry as shape
from shapely.geometry import LineString


class Obstacle:
    def __init__(self):
        self.id = -1 # actor id
        self.vx = 0.0 # velocity in x direction
        self.vy = 0.0 # velocity in y direction
        self.vz = 0.0 # velocity in z direction
        self.ros_transform = None # transform of the obstacle in ROS coordinate
        self.carla_transform = None # transform of the obstacle in Carla world coordinate
        self.bbox = None # Bounding box w.r.t ego vehicle's local frame
    
    def get_bbox_center(self, vertices):
        """
        Get the center of the bounding box
        """
        x = [v.x for v in vertices]
        y = [v.y for v in vertices]
        z = [v.z for v in vertices]

        center = Point()
        center.x, center.y, center.z = np.mean(x), np.mean(y), np.mean(z)
        return center

class MyLocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is
    generated on-the-fly. The low-level motion of the vehicle is computed by using two PID
    controllers, one is used for the lateral control and the other for the longitudinal
    control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, role_name, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param role_name: name of the actor
        :param opt_dict: dictionary of arguments with the following semantics:

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal
                                         PID controller
                                         {'K_P':, 'K_D':, 'K_I'}
        """
        self.target_route_point = None
        self._current_waypoint = None
        self._vehicle_controller = None
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._avoid_buffer = deque(maxlen=self._buffer_size)
        self._avoid_flag = False
        self._vehicle_yaw = None
        self._current_speed = None
        self._current_pose = None
        self._obstacles = []

        # get world and map for finding actors and waypoints
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()
        self.map = self.world.get_map()        

        self._target_point_publisher = rospy.Publisher(
            "/next_target", PointStamped, queue_size=1)

        rospy.wait_for_service('/carla_waypoint_publisher/{}/get_waypoint'.format(role_name))
        self._get_waypoint_client = rospy.ServiceProxy(
            '/carla_waypoint_publisher/{}/get_waypoint'.format(role_name), GetWaypoint)

        # initializing controller
        self._init_controller(opt_dict)

    def get_obstacles(self, location, range):
        """
        Get a list of obstacles that are located within a certain distance from the location.
        
        :param      location: queried location
        :param      range: search distance from the queried location
        :type       location: geometry_msgs/Point
        :type       range: float or double
        :return:    None
        :rtype:     None
        """
        self._obstacles = []
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if "role_name" in actor.attributes:
                if actor.attributes["role_name"] == 'autopilot' or actor.attributes["role_name"] == "static":
                    carla_transform = actor.get_transform()
                    ros_transform = trans.carla_transform_to_ros_pose(carla_transform)
                    x = ros_transform.position.x
                    y = ros_transform.position.y
                    z = ros_transform.position.z 
                    distance = math.sqrt((x-location.x)**2 + (y-location.y)**2)
                    if distance < range:
                        # print("obs distance: {}").format(distance)
                        ob = Obstacle()
                        ob.id = actor.id
                        ob.carla_transform = carla_transform
                        ob.ros_transform = ros_transform
                        ob.vx = actor.get_velocity().x
                        ob.vy = actor.get_velocity().y
                        ob.vz = actor.get_velocity().z
                        ob.bbox = actor.bounding_box # in local frame
                        # print("x: {}, y: {}, z:{}").format(x, y, z)
                        # print("bbox x:{} y:{} z:{} ext: {} {} {}".format(ob.bbox.location.x, ob.bbox.location.y, ob.bbox.location.z, ob.bbox.extent.x, ob.bbox.extent.y, ob.bbox.extent.z))
                        self._obstacles.append(ob)

    def check_obstacle(self, point, obstacle):
        """
        Check whether a point is inside the bounding box of the obstacle

        :param      point: a location to check the collision (in ROS frame)
        :param      obstacle: an obstacle for collision check
        :type       point: geometry_msgs/Point
        :type       obstacle: object Obstacle
        :return:    true or false
        :rtype:     boolean   
        """
        carla_location = carla.Location()
        carla_location.x = point.x
        carla_location.y = -point.y
        carla_location.z = point.z
        
        vertices = obstacle.bbox.get_world_vertices(obstacle.carla_transform)
        
        vx = [v.x for v in vertices]
        vy = [v.y for v in vertices]
        vz = [v.z for v in vertices]
        return carla_location.x >= min(vx) and carla_location.x <= max(vx) \
                and carla_location.y >= min(vy) and carla_location.y <= max(vy) \
                and carla_location.z >= min(vz) and carla_location.z <= max(vz) 

    
    def check_obstacle_fov(self, pose, fov_angle, fov_radius, target_speed):
        """
        Check whether an obstacle is within the field of view of the ego vehicle

        :param      pose: ego vehicle pose
        :param      obstacle: an obstacle for collision check
        :param      fov_angle: field of view angle
        :param      fov_radius: field of view radius
        :type       pose: geometry_msgs/Pose
        :type       obstacle: object Obstacle
        :type       fov_angle: float or double
        :type       fov_radius: float or double
        :return:    true or false
        :rtype:     boolean
        """
        # Helper function to calculate 2D angle between car and a 2D point
        def calculate_2d_angle(x1, y1, x2, y2):
            return math.atan2(y2 - y1, x2 - x1)

        point = pose.position
        # convert the ego vehicle postion to ROS frame
        carla_location = carla.Location()
        carla_location.x = point.x
        carla_location.y = -point.y
        carla_location.z = point.z
        
        left_lane, right_lane, left_angle, right_angle = self.get_left_right_lanes_waypoint_pose(pose.position, fov_radius)

        # Calculate the cosine of half the FOV angle for comparison
        cos_half_fov = math.cos(fov_angle / 2.0)
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )
        _, _, vehicle_yaw = euler_from_quaternion(quaternion=quaternion)
        print("vehicle_yaw: {}".format(vehicle_yaw))

        obstacle_angles = []
        obstacle_locations = []
        hit = False
        for ob in self._obstacles:
            # get the vertices of the bounding box
            vertices = ob.bbox.get_world_vertices(ob.carla_transform)
            # Check if the ray hits the obstacle  
            for v in vertices:
                angle = calculate_2d_angle(carla_location.x, carla_location.y, v.x, v.y)

                dot_product = math.cos(angle - vehicle_yaw)

                # check if the obstacle is within the FOV
                if dot_product >= cos_half_fov:
                    # set hit to true if the obstacle is in the same lane as the ego vehicle
                    center = ob.get_bbox_center(vertices)
                    center.y = -center.y
                    print("\x1b[6;30;33m------ Angles-----\x1b[0m")
                    print(left_angle, angle, right_angle)                    
                    if left_angle < angle < right_angle:
                        hit = True
                        self._avoid_flag = True
                        obstacle_locations.append(center)
                    obstacle_angles.append(self.normalize_angle(math.floor(angle)))

        # Create a range of angles spanning FOV with step 1 in degrees
        lower_bound = int(self.normalize_angle(vehicle_yaw - fov_angle / 2.0))
        upper_bound = int(self.normalize_angle(vehicle_yaw + fov_angle / 2.0))+1
        fov_range_rad = list(range(lower_bound, upper_bound, int(math.ceil(math.radians(4)))))

        # Remove the angles which hit the obstacle
        safe_angles_fov = [angle for angle in fov_range_rad \
                              if all(abs(angle - obstacle_angle) > 4 for obstacle_angle in obstacle_angles)]
        
        # default lane change is false
        target_route_point = self._waypoint_buffer[0]
        print("\x1b[6;30;33m------target_rout_points default------\x1b[0m")
        print(target_route_point)
        # print(target_route_point)

        # obstacle is in the same lane, find the best lane change
        if hit:
            print("\x1b[6;30;33m------Obstacle in the same lane------\x1b[0m")
            ego_vehicle_waypoint = self.get_waypoint(pose.position)
           
            print("left_lane: {}".format(left_lane))
            # print("right_lane: {}".format(right_lane))
            print("ego_vehicle_waypoint: {}".format(ego_vehicle_waypoint))
            safe_lanes = self.get_safe_lanes(ego_vehicle_waypoint)
            # remove the angles which lead to incorrect lane change
            print("\x1b[6;30;33m------Safe Lanes------\x1b[0m")
            print(safe_lanes)
            if len(safe_angles_fov) > 0:
                for angle in safe_angles_fov:
                    # find the x,y coordinates of the point on the circle and find the corresponding waypoint
                    x = fov_radius * math.cos(angle) + carla_location.x
                    y = fov_radius * math.sin(angle) + carla_location.y
                    z = carla_location.z
                    # find the waypoint

                    if 0< angle < right_angle:
                        # in the right lane and its not safe to change to the right lane
                        if safe_lanes.count('right') == 0:
                            safe_angles_fov.remove(angle)                        

                    elif 0 > angle > left_angle:
                        # in the left lane
                        if safe_lanes.count('left') == 0:
                            safe_angles_fov.remove(angle)
            # now find the next waypoints of safe lane changes

            # find the closest obstacle to the ego vehicle in obstacle_locations
            closest_obstacle = None
            min_distance = 1000000
            for ob in obstacle_locations:
                distance = math.sqrt((ob.x-carla_location.x)**2 + (ob.y-carla_location.y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = ob
            print("\x1b[6;30;33m------Closest Obstacle------\x1b[0m")
            print(obstacle_locations)
            if closest_obstacle is not None:
                if len(safe_lanes) > 0:
                    left, right, _, _ = self.get_left_right_lanes_waypoint_pose(closest_obstacle, fov_radius)
                    obstacle_end_point = self.make_new_pose(pose, self.get_forward_next_waypoint_pose(closest_obstacle, 20.0))
                    if 'right' in safe_lanes:
                        new_pose = self.make_new_pose(pose, right)
                        if target_route_point.position.x != new_pose.position.x \
                            and target_route_point.position.y != new_pose.position.y:
                            target_route_point = new_pose
                            self._waypoint_buffer.appendleft(obstacle_end_point)
                            right_forward = self.make_new_pose(pose, self.get_forward_next_waypoint_pose(right, 20.0))
                            self._waypoint_buffer.appendleft(right_forward)
                            self._waypoint_buffer.appendleft(new_pose)                    
                            print("\x1b[6;30;33m------target_rout_points right lane------\x1b[0m")
                            print(target_route_point)
                    elif 'left' in safe_lanes:
                        new_pose = self.make_new_pose(pose, left)
                        if target_route_point.position.x != new_pose.position.x \
                            and target_route_point.position.y != new_pose.position.y:
                            target_route_point = new_pose
                            self._waypoint_buffer.appendleft(obstacle_end_point)
                            left_forward = self.make_new_pose(pose, self.get_forward_next_waypoint_pose(left, 20.0))
                            self._waypoint_buffer.appendleft(left_forward)
                            self._waypoint_buffer.appendleft(new_pose)
                            print("\x1b[6;30;33m------target_rout_points left lane------\x1b[0m")
                            print(target_route_point)
                    else:
                        # no safe lane change, decelerate
                        target_speed = min(target_speed, 20.0)

        return target_route_point, target_speed
    

    def normalize_angle(self, angle):
        while angle >= math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def make_new_pose(self, ego_vehicle_pose, new_lane_location):
        """
        Make a new pose for the ego vehicle to move to the new lane
        """
        new_pose = Pose()
        new_pose.position.x = new_lane_location.x
        new_pose.position.y = new_lane_location.y
        new_pose.position.z = new_lane_location.z
        new_pose.orientation.x = ego_vehicle_pose.orientation.x
        new_pose.orientation.y = ego_vehicle_pose.orientation.y
        new_pose.orientation.z = ego_vehicle_pose.orientation.z
        new_pose.orientation.w = ego_vehicle_pose.orientation.w
        return new_pose
    
    def get_safe_lanes(self, ego_vehicle_waypoint):
        safe_lanes =[]
        if ego_vehicle_waypoint.right_lane_marking.type == "Broken":
            safe_lanes.append('right')
        if ego_vehicle_waypoint.left_lane_marking.type == "Broken":
            safe_lanes.append('left') 
        return safe_lanes


    
    # def check_obstacle_front(self, pose, lane_width, fov_radius):
    #     # fov_angle = math.atan2(lane_width/2, fov_radius)
    #     fov_angle = np.radians(120)
    #     for ob in self._obstacles:
    #         hit, angles_outside_fov = self.check_obstacle_fov(pose=pose,obstacle=ob, fov_angle=fov_angle, fov_radius=fov_radius)
    #         if hit:
    #             return True, ob
    #     return False, None
         
    # def obstacle_manuever(self, pose, current_speed, lane_width, fov_radius, target_speed):
    #     # tif there is an obstacle in the front move to the right
    #     hit, ob =  self.check_obstacle_front(pose, lane_width, fov_radius=10.0)
    #     # target waypoint
    #     self.target_route_point = self._waypoint_buffer[0]
    #     if hit:
    #         print("\x1b[6;30;33m------Obstacle in the front------\x1b[0m")
    #         vertices = ob.bbox.get_world_vertices(ob.carla_transform)
    #         center = ob.get_bbox_center(vertices)
    #         # coordinates of the left and right lane markings where the vehicle need to move
    #         right_future, left_future = self.get_coordinate_lanemarking(center)
    #         right_current, left_current = self.get_coordinate_lanemarking(pose.position)
    #         # check if both right lane markings are free
    #         available_right, available_left = True, True
    #         for ob in self._obstacles:
    #             if self.check_obstacle(right_future, ob) or self.check_obstacle(right_current, ob):
    #                 available_right = False
    #             if self.check_obstacle(left_future, ob) or self.check_obstacle(left_current, ob):
    #                 available_left = False
            
    #         if available_right:
    #             print("\x1b[6;30;33m------Move to the right------\x1b[0m")
    #             # append right_future to the top of waypoint_buffer to move to the right
    #             right_future.y = -right_future.y
    #             print("right_future: {}".format(right_future))
    #             next_waypoint = self.get_waypoint(right_future)
    #             if next_waypoint is not None:
    #                 print("current waypoint: {}".format(self.target_route_point.position))
    #                 self.target_route_point = next_waypoint.pose
    #                 print("next waypoint: {}".format(next_waypoint.pose.position))
                  
    #             else:
    #                 # If there is no right lane, decelerate
    #                 target_speed = min(target_speed, 20.0)

    #         elif available_left:
    #             print("move to the left")
    #             left_future.y = -left_future.y
    #             # append left_future to the top of waypoint_buffer to move to the left
    #             next_waypoint = self.get_waypoint(left_future)
    #             if next_waypoint is not None:
    #                 self.target_route_point = next_waypoint.pose
    #             else:
    #                 # If there is no left lane, decelerate
    #                 target_speed = min(target_speed, 20.0)
    #         else:
    #             print("no lane change")
    #             # If there is no lane change, decelerate
    #             target_speed = min(target_speed, 20.0)
  
             
    #     # publish target point
    #     target_point = PointStamped()
    #     target_point.header.frame_id = "map"
    #     target_point.point.x = self.target_route_point.position.x
    #     target_point.point.y = self.target_route_point.position.y
    #     target_point.point.z = self.target_route_point.position.z
    #     self._target_point_publisher.publish(target_point)
        
    #     # move using PID controllers
    #     control = self._vehicle_controller.run_step(
    #         target_speed, current_speed, pose, self.target_route_point)
        
        
    #     return control


    def get_left_right_lanes_waypoint_pose(self, position, fov_radius):
        """
        Helper to get adjacent waypoint 2D coordinates of the left and right lanes. 
        with respect to the closest waypoint
        
        :param      position: queried position
        :type       position: geometry_msgs/Point
        :return:    left and right waypoint in numpy array
        :rtype:     tuple of geometry_msgs/Point (left), geometry_msgs/Point (right)
        """
        # get waypoints along road
        current_waypoint = self.get_waypoint(position)
        waypoint_xodr = self.map.get_waypoint_xodr(current_waypoint.road_id, current_waypoint.lane_id, current_waypoint.s)
        
        # find two orthonormal vectors to the direction of the lane
        yaw = math.pi - waypoint_xodr.transform.rotation.yaw * math.pi / 180.0
        v = np.array([1.0, math.tan(yaw)])
        norm_v = v / np.linalg.norm(v)
        right_v = np.array([-norm_v[1], norm_v[0]])
        left_v = np.array([norm_v[1], -norm_v[0]])
        
        # find two points that are on the left and right lanes
        width = current_waypoint.lane_width*1.5
        left_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + width * left_v
        right_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + width * right_v
        ros_left_waypoint = Point()
        ros_right_waypoint = Point()
        ros_left_waypoint.x = left_waypoint[0]
        ros_left_waypoint.y = left_waypoint[1]
        ros_right_waypoint.x = right_waypoint[0]
        ros_right_waypoint.y = right_waypoint[1]

        # get lane markings angles
        left_angle = 3*math.pi/2 - math.acos(width/(2*fov_radius))
        right_angle = math.pi/2 + math.acos(width/(2*fov_radius))

        left_angle = self.normalize_angle(left_angle)
        right_angle = self.normalize_angle(right_angle)

        return ros_left_waypoint, ros_right_waypoint, left_angle, right_angle
    
    def get_forward_next_waypoint_pose(self, position, forward_distance):
         # get waypoints along road
        current_waypoint = self.get_waypoint(position)
        waypoint_xodr = self.map.get_waypoint_xodr(current_waypoint.road_id, current_waypoint.lane_id, current_waypoint.s)
        
        # find two orthonormal vectors to the direction of the lane
        yaw = math.pi - waypoint_xodr.transform.rotation.yaw * math.pi / 180.0
        v = np.array([1.0, math.tan(yaw)])
        norm_v = v / np.linalg.norm(v)
        right_v = np.array([-norm_v[1], norm_v[0]])
        left_v = np.array([norm_v[1], -norm_v[0]])
        
        # Find current waypoint
        current_waypoint_pos = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y])

        # Calculate forward translation based on the current direction (lane direction)
        forward_point = current_waypoint_pos + forward_distance * norm_v
        ros_forward_point = Point()
        ros_forward_point.x = forward_point[0]
        ros_forward_point.y = forward_point[1]

        return ros_forward_point

       
                

    def get_coordinate_lanemarking(self, position):
        """
        Helper to get adjacent waypoint 2D coordinates of the left and right lane markings 
        with respect to the closest waypoint
        
        :param      position: queried position
        :type       position: geometry_msgs/Point
        :return:    left and right waypoint in numpy array
        :rtype:     tuple of geometry_msgs/Point (left), geometry_msgs/Point (right)
        """
        # get waypoints along road
        current_waypoint = self.get_waypoint(position)
        waypoint_xodr = self.map.get_waypoint_xodr(current_waypoint.road_id, current_waypoint.lane_id, current_waypoint.s)
        
        # find two orthonormal vectors to the direction of the lane
        yaw = math.pi - waypoint_xodr.transform.rotation.yaw * math.pi / 180.0
        v = np.array([1.0, math.tan(yaw)])
        norm_v = v / np.linalg.norm(v)
        right_v = np.array([-norm_v[1], norm_v[0]])
        left_v = np.array([norm_v[1], -norm_v[0]])
        
        # find two points that are on the left and right lane markings
        half_width = current_waypoint.lane_width / 2.0
        left_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + half_width * left_v
        right_waypoint = np.array([current_waypoint.pose.position.x, current_waypoint.pose.position.y]) + half_width * right_v
        ros_left_waypoint = Point()
        ros_right_waypoint = Point()
        ros_left_waypoint.x = left_waypoint[0]
        ros_left_waypoint.y = left_waypoint[1]
        ros_right_waypoint.x = right_waypoint[0]
        ros_right_waypoint.y = right_waypoint[1]
        return ros_left_waypoint, ros_right_waypoint

    def get_waypoint(self, location):
        """
        Helper to get waypoint from a ros service
        """
        try:
            response = self._get_waypoint_client(location)
            return response.waypoint
        except (rospy.ServiceException, rospy.ROSInterruptException) as e:
            if not rospy.is_shutdown:
                rospy.logwarn("Service call failed: {}".format(e))

    def odometry_updated(self, odo):
        """
        Callback on new odometry
        """
        self._current_speed = math.sqrt(odo.twist.twist.linear.x ** 2 +
                                        odo.twist.twist.linear.y ** 2 +
                                        odo.twist.twist.linear.z ** 2) * 3.6

        self._current_pose = odo.pose.pose
        quaternion = (
            odo.pose.pose.orientation.x,
            odo.pose.pose.orientation.y,
            odo.pose.pose.orientation.z,
            odo.pose.pose.orientation.w
        )
        _, _, self._vehicle_yaw = euler_from_quaternion(quaternion)

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4}
        args_longitudinal_dict = {
            'K_P': 0.2,
            'K_D': 0.05,
            'K_I': 0.1}

        # parameters overload
        if opt_dict:
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._vehicle_controller = VehiclePIDController(args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

    def set_global_plan(self, current_plan):
        """
        set a global plan to follow
        """
        self.target_route_point = None
        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem.pose)

    def run_step(self, target_speed, current_speed, current_pose):
        """
        Execute one step of local planning which involves running the longitudinal
        and lateral PID controllers to follow the waypoints trajectory.
        """
        if not self._waypoint_buffer and not self._waypoints_queue:
            control = CarlaEgoVehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            rospy.loginfo("Route finished.")
            return control, True

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self.get_waypoint(current_pose.position)

        # Field of View (FOV) obstacle detection parameters
        fov_angle = np.radians(180)  # Set the FOV angle (e.g., 180 degrees)
        fov_radius = self._current_waypoint.lane_width*4 # Set the FOV radius (e.g., 10 meters)
        # get a list of obstacles surrounding the ego vehicle
        self.get_obstacles(current_pose.position, fov_radius)

        # # # Example 1: get two waypoints on the left and right lane marking w.r.t current pose
        # left, right = self.get_coordinate_lanemarking(current_pose.position)
        # print("\x1b[6;30;33m------Example 1------\x1b[0m")
        # print("Left: {}, {}; right: {}, {}".format(left.x, left.y, right.x, right.y))

        # # find if there are obstacles in current lane
        # left_lane, right_lane = self.get_coordinate_lanemarking(current_pose.position)
        # # left_obstacle = False
        # # right_obstacle = False
        # # current_lane_obstacle = False

        # # for ob in self._obstacles:
        # #     # Check for obstacles in the left lane
        # #     if self.check_obstacle(left_lane, ob):
        # #         left_obstacle = True
        # #     # Check for obstacles in the right lane
        # #     if self.check_obstacle(right_lane, ob):
        # #         right_obstacle = True
      

        # if current_lane_obstacle:
        #     print("-- no lane change --")
        #     # If there is an obstacle in the current lane, check for the left and right lanes
        #     if left_obstacle and right_obstacle:
        #         # If there are obstacles in both lanes, stop the vehicle
        #          target_speed = min(target_speed, 50.0)  # Adjust the minimum speed as needed
        #     elif left_obstacle:
        #         # If there is an obstacle in the left lane, switch to the right lane
        #         # find the next waypoint in the right lane
        #         next_waypoint = self._current_waypoint.get_right_lane()
        #         if next_waypoint is not None:
        #             self._waypoints_queue.appendleft(next_waypoint.pose)
        #             self._waypoint_buffer.appendleft(next_waypoint.pose)
        #         else:
        #             # If there is no right lane, decelerate
        #             target_speed = min(target_speed, 50.0)  # Adjust the minimum speed as needed                    
        #     elif right_obstacle:
        #         # If there is an obstacle in the right lane, switch to the left lane
        #         # find the next waypoint in the left lane
        #         next_waypoint = self._current_waypoint.get_left_lane()
        #         if next_waypoint is not None:
        #             self._waypoints_queue.appendleft(next_waypoint.pose)
        #             self._waypoint_buffer.appendleft(next_waypoint.pose)
        #         else:
        #             # If there is no left lane, decelerate
        #             target_speed = min(target_speed, 50.0)  # Adjust the minimum speed as needed
        

        # Field of View (FOV) obstacle detection
       
        # car_fov_obstacle = False
        # print("\x1b[6;30;33m------Collision Check------\x1b[0m")
        # for ob in self._obstacles:
        #     hit, angles_outside_fov = self.check_obstacle_fov(pose=current_pose,obstacle=ob, fov_angle=fov_angle, fov_radius=fov_radius)
        #     print("id: {}, collision: {}".format(ob.id, hit))
        #     print("angles_outside_fov: {}".format(angles_outside_fov))

        # 
        # control = self.obstacle_manuever(pose=current_pose, current_speed=current_speed, lane_width=self._current_waypoint.lane_width, fov_radius=fov_radius, target_speed=target_speed)
        # # for ob in self._obstacles:
        #     if self.check_obstacle_fov(current_pose.position, ob, fov_angle, fov_radius):
        #         car_fov_obstacle = True
        #         break

        # Calculate safe angles considering obstacles in the FOV
        # safe_angles = self.get_clear_angles(current_pose=current_pose, fov_angle=fov_angle)
        # print("\xlib[6;30;33m------Safe Angles------\x1b[0m")
        # print("Safe angles: {}".format(safe_angles))
        # if car_fov_obstacle:
        #     clear_angles = self.get_clear_angles(current_pose, fov_angle)
        #     for start, end in clear_angles:
        #         if not any(start < angle < end for angle in safe_angles):
        #             safe_angles.append((start, end))
        # else:
        #     safe_angles = self.get_clear_angles(current_pose, fov_angle)

        # print("Safe angles: {}".format(safe_angles))
        # Example 2: check obstacle collision
        # print("\x1b[6;30;33m------Example 2------\x1b[0m")
        # point = Point()
        # point.x = 100.0
        # point.y = 100.0
        # point.z = 1.5
        # for ob in self._obstacles:
        #     print("id: {}, collision: {}".format(ob.id, self.check_obstacle(point, ob)))
        
        target_rout_point, target_speed = self.check_obstacle_fov(current_pose, fov_angle, fov_radius, target_speed)

        self.target_route_point = target_rout_point
        # publish target point
        target_point = PointStamped()
        target_point.header.frame_id = "map"
        target_point.point.x = self.target_route_point.position.x
        target_point.point.y = self.target_route_point.position.y
        target_point.point.z = self.target_route_point.position.z
        self._target_point_publisher.publish(target_point)
        
        # move using PID controllers
        control = self._vehicle_controller.run_step(
            target_speed, current_speed, current_pose, self.target_route_point)
        
       
        # purge the queue of obsolete waypoints
        max_index = -1

        sampling_radius = target_speed * 1 / 3.6  # 1 seconds horizon
        min_distance = sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        for i, route_point in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    route_point, current_pose.position) < min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        return control, False