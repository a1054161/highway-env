from gym.envs.registration import register
from gym import GoalEnv
import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """
    REWARD_WEIGHTS: ndarray = np.array([1, 0.3, 0, 0, 0.02, 0.02])  
    SUCCESS_GOAL_REWARD: float = 0.06  
    STEERING_RANGE: float = np.deg2rad(45) 
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": np.array([1, 0.3, 0, 0, 0.02, 0.02]),
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1
        })
        return config

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 15) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, [i*20, 0], 2*np.pi*self.np_random.rand(), 0)
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        
        """
        penalty = 0             #進入懲罰區懲罰
        #penalty_angle = 0       #入庫角度過大懲罰
        penalty_value = -0.6      #進入懲罰區懲罰值
        #penalty_angle_value = 0 #入庫角度過大懲罰值
        a0 = (achieved_goal[0], achieved_goal[1]) #中心點
        a1 = (achieved_goal[0]-0.025, achieved_goal[1]-0.01) #左上點
        a2 = (achieved_goal[0]-0.025, achieved_goal[1]+0.01) #左下點
        a3 = (achieved_goal[0]+0.025, achieved_goal[1]+0.01) #右下點
        a4 = (achieved_goal[0]+0.025, achieved_goal[1]-0.01) #右上點
        a5 = (achieved_goal[0], achieved_goal[1]+0.01) #側下點
        a6 = (achieved_goal[0], achieved_goal[1]-0.01) #側上點
        a7 = (achieved_goal[0]+0.025, achieved_goal[1])#右側點
        a8 = (achieved_goal[0]-0.025, achieved_goal[1])#左側點
        dx = desired_goal[0]
        dy = desired_goal[1]
        #.astype(np.float64)
        #
        theta1 = np.degrees(np.arccos(achieved_goal[4])) 
        theta2 = np.degrees(np.arcsin(achieved_goal[5]))
        rotate_theta = 0
        
        if (theta1 >[0]).all() and (theta2 > [0]).all():
            #print("2")
            rotate_theta = theta1
        else:
            rotate_theta = 180 - theta1
            #print("1")
        #print(theta1, rotate_theta)

        rotate_radians = np.radians(rotate_theta)

        
        rx1 = (a1[0]-a0[0])*(np.cos(rotate_radians))-(a1[1]-a0[1])*(np.sin(rotate_radians))+a0[0]
        ry1 = (a1[1]-a0[1])*(np.cos(rotate_radians))+(a1[0]-a0[0])*(np.sin(rotate_radians))+a0[1]
        rx2 = (a2[0]-a0[0])*(np.cos(rotate_radians))-(a2[1]-a0[1])*(np.sin(rotate_radians))+a0[0]
        ry2 = (a2[1]-a0[1])*(np.cos(rotate_radians))+(a2[0]-a0[0])*(np.sin(rotate_radians))+a0[1]
        rx3 = (a3[0]-a0[0])*(np.cos(rotate_radians))-(a3[1]-a0[1])*(np.sin(rotate_radians))+a0[0]
        ry3 = (a3[1]-a0[1])*(np.cos(rotate_radians))+(a3[0]-a0[0])*(np.sin(rotate_radians))+a0[1]
        rx4 = (a4[0]-a0[0])*(np.cos(rotate_radians))-(a4[1]-a0[1])*(np.sin(rotate_radians))+a0[0]
        ry4 = (a4[1]-a0[1])*(np.cos(rotate_radians))+(a4[0]-a0[0])*(np.sin(rotate_radians))+a0[1]
        rx5 = (a5[0]-a0[0])*(np.cos(rotate_radians))-(a5[1]-a0[1])*(np.sin(rotate_radians))+a0[0]
        ry5 = (a5[1]-a0[1])*(np.cos(rotate_radians))+(a5[0]-a0[0])*(np.sin(rotate_radians))+a0[1]
        rx6 = (a6[0]-a0[0])*(np.cos(rotate_radians))-(a6[1]-a0[1])*(np.sin(rotate_radians))+a0[0]
        ry6 = (a6[1]-a0[1])*(np.cos(rotate_radians))+(a6[0]-a0[0])*(np.sin(rotate_radians))+a0[1]
        
        b1 = [0.28]
        b2 = [-0.32]
        b3 = [0.18]
        b4 =[-0.18]
        if (rx1 > b1).all() or (rx2 > b1).all() or (rx3 > b1).all() or (rx4 > b1).all() or (rx5 > b1).all() or (rx6 > b1).all():
            penalty = penalty_value      #右半
        elif (rx1 < b2).all() or (rx2 < b2).all() or (rx3 < b2).all() or (rx4 < b2).all() or (rx5 < b2).all() or (rx6 < b2).all():
            penalty = penalty_value      #左半
        elif (ry1 > b3).all or (ry2 > b3).all() or (ry3 > b3).all() or (ry4 > b3).all() or (ry5 > b3).all() or (ry6 > b3).all():
            penalty = penalty_value      #下半
        elif (ry1 < b4).all() or (ry2 < b4).all() or (ry3 < b4).all() or (ry4 < b4).all() or (ry5 < b4).all() or (ry6 < b4).all():
            penalty = penalty_value      #上半
        elif (dx+[0.02] < rx1 < [0.28]).all() and (dy-[0.045] < ry1 < dy+[0.045]).all():   
            penalty = penalty_value      #第一個點在停車場右邊區域
        elif (dx+[0.02] < rx2 < [0.28]).all() and (dy-[0.045] < ry2 < dy+[0.045]).all():
            penalty = penalty_value     #第二個點在停車場右邊區域
        elif (dx+[0.02] < rx3 < [0.28]).all() and (dy-[0.045] < ry3 < dy+[0.045]).all():
            penalty = penalty_value      #第三個點在停車場右邊區域
        elif (dx+[0.02] < rx4 < [0.28]).all() and (dy-[0.045] < ry4 < dy+[0.045]).all():
            penalty = penalty_value      #第四個點在停車場右邊區域
        elif (dx+[0.02] < rx5 < [0.28]).all() and (dy-[0.045] < ry5 < dy+[0.045]).all():
            penalty = penalty_value      #第五個點在停車場右邊區域
        elif (dx+[0.02] < rx6 < [0.28]).all() and (dy-[0.045] < ry6 < dy+[0.045]).all():
            penalty = penalty_value      #第六個點在停車場右邊區域
        elif ([-0.32] < rx1 < dx-[0.02]).all() and (dy-[0.045] < ry1 < dy+[0.045]).all():
            penalty = penalty_value      #第一個點在停車場左邊區域
        elif ([-0.32] < rx2 < dx-[0.02]).all() and (dy-[0.045] < ry2 < dy+[0.045]).all():
            penalty = penalty_value      #第二個點在停車場左邊區域
        elif ([-0.32] < rx3 < dx-[0.02]).all() and (dy-[0.045] < ry3 < dy+[0.045]).all():
            penalty = penalty_value      #第三個點在停車場左邊區域
        elif ([-0.32] < rx4 < dx-[0.02]).all() and (dy-[0.045] < ry4 < dy+[0.045]).all():
            penalty = penalty_value      #第四個點在停車場左邊區域
        elif ([-0.32] < rx5 < dx-[0.02]).all() and (dy-[0.045] < ry5 < dy+[0.045]).all():
            penalty = penalty_value      #第五個點在停車場左邊區域
        elif ([-0.32] < rx6 < dx-[0.02]).all() and (dy-[0.045] < ry6 < dy+[0.045]).all():
            penalty = penalty_value      #第六個點在停車場左邊區域
        elif ([-0.32] < rx1 < [0.28]).all() and (-dy-[0.045] < ry1 < -dy+[0.045]).all():
            penalty = penalty_value      #第一個點在停車場相反區域
        elif ([-0.32] < rx2 < [0.28]).all() and (-dy-[0.045] < ry2 < -dy+[0.045]).all():
            penalty = penalty_value      #第二個點在停車場相反區域
        elif ([-0.32] < rx3 < [0.28]).all() and (-dy-[0.045] < ry3 < -dy+[0.045]).all():
            penalty = penalty_value      #第三個點在停車場相反區域
        elif ([-0.32] < rx4 < [0.28]).all() and (-dy-[0.045] < ry4 < -dy+[0.045]).all():
            penalty = penalty_value      #第四個點在停車場相反區域
        elif ([-0.32] < rx5 < [0.28]).all() and (-dy-[0.045] < ry5 < -dy+[0.045]).all():
            penalty = penalty_value      #第五個點在停車場相反區域
        elif ([-0.32] < rx6 < [0.28]).all() and (-dy-[0.045] < ry6 < -dy+[0.045]).all():
            penalty = penalty_value      #第六個點在停車場相反區域      
        
        c = - np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p) + penalty
        #print("c",c)
        return c
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.config["reward_weights"]), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        return sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {})
                     for agent_obs in obs)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return time or crashed or success


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


register(
    id='parking-v0',
    entry_point='highway_env.envs:ParkingEnv',
)

register(
    id='parking-ActionRepeat-v0',
    entry_point='highway_env.envs:ParkingEnvActionRepeat'
)
