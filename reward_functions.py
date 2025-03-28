"""
激光车辆强化学习 - 奖励函数
包含多级奖励机制设计，用于训练智能体
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Any

class RewardCalculator:
    """奖励计算器类，负责计算智能体获得的奖励"""
    
    def __init__(self):
        """
        多维度奖励计算器
        """
        # 奖励权重配置 - 默认值
        self.reward_weights = {
            'survival': 0.05,      # 生存奖励
            'movement': 0.03,      # 移动奖励
            'distance': 0.1,       # 距离目标奖励
            'target': 2.0,         # 击中目标奖励
            'collision': -5.0,     # 碰撞惩罚
            'mine': -10.0,         # 触发雷区惩罚
            'out_of_bounds': -5.0, # 出界惩罚
            'efficiency': 0.02,    # 能量效率奖励
            'laser': 0.1,          # 激光使用奖励
            'exploration': 0.05,   # 探索奖励
            'stability': 0.02      # 稳定性奖励
        }
        
        # 历史信息
        self.prev_position = None
        self.prev_orientation = None
        self.prev_velocity = None
        self.visited_areas = np.zeros((20, 20))  # 20x20网格记录已探索区域
        self.grid_size = 0.5  # 每个网格单元的大小(米)
        
        # 追踪信息
        self.laser_hits = 0
        self.total_efficiency = 0.0
    
    def compute_reward(self, vehicle, arena, action, step, max_steps, difficulty) -> Tuple[float, Dict[str, float]]:
        """
        计算当前步骤的多维度奖励
        
        参数:
            vehicle: 车辆对象
            arena: 战斗场景
            action: 当前执行的动作
            step: 当前步数
            max_steps: 最大步数
            difficulty: 当前难度(0.1-1.0)
            
        返回:
            float: 总奖励值
            Dict[str, float]: 各维度奖励值
        """
        # 获取车辆状态
        position, orientation = vehicle.get_position_and_orientation()
        velocity = vehicle.get_velocity()
        angular_velocity = vehicle.get_angular_velocity()
        
        # 各维度奖励初始化
        reward_components = {
            'survival': 0.0,
            'movement': 0.0,
            'target': 0.0,
            'laser': 0.0,
            'collision': 0.0,
            'mine': 0.0,
            'efficiency': 0.0,
            'exploration': 0.0,
            'stability': 0.0
        }
        
        # 1. 生存奖励 - 每个时间步都有小奖励
        reward_components['survival'] = self.reward_weights['survival'] * (1 + 0.5 * difficulty)
        
        # 2. 移动奖励 - 鼓励车辆移动
        if self.prev_position is not None:
            distance_moved = np.linalg.norm(np.array(position[:2]) - np.array(self.prev_position[:2]))
            if distance_moved > 0.01:  # 需要移动足够距离
                reward_components['movement'] = self.reward_weights['movement'] * distance_moved
        
        # 3. 碰撞惩罚
        if vehicle.check_collision():
            reward_components['collision'] = self.reward_weights['collision']
        
        # 4. 雷区惩罚
        min_mine_distance = self._get_min_mine_distance(vehicle, arena)
        if min_mine_distance < arena.mine_radius:
            reward_components['mine'] = self.reward_weights['mine']
        elif min_mine_distance < arena.mine_radius * 2:
            # 接近雷区给予警告惩罚
            reward_components['mine'] = self.reward_weights['mine'] * 0.5 * (1 - min_mine_distance / (arena.mine_radius * 2))
        
        # 5. 出界惩罚
        if abs(position[0]) > 4.5 or abs(position[1]) > 4.5:
            reward_components['collision'] += self.reward_weights['out_of_bounds']
        
        # 6. 能量效率奖励 - 鼓励优化的动作
        action_magnitude = np.linalg.norm(action)
        velocity_magnitude = np.linalg.norm(velocity[:2])  # 只考虑x-y平面速度
        
        # 高速度/低动作幅度比值越高越好
        if action_magnitude > 0.1:  # 防止除以接近零的值
            efficiency = velocity_magnitude / action_magnitude
            reward_components['efficiency'] = self.reward_weights['efficiency'] * efficiency
            self.total_efficiency += efficiency
        
        # 7. 激光使用奖励 - 如果车辆使用激光并击中目标
        if hasattr(vehicle, 'laser_active') and vehicle.laser_active:
            laser_hit_something = vehicle.check_laser_hit()
            if laser_hit_something:
                reward_components['laser'] = self.reward_weights['laser'] * (1 + difficulty)
                self.laser_hits += 1
        
        # 8. 探索奖励 - 鼓励车辆探索未访问区域
        grid_x = int((position[0] + 5) / self.grid_size)
        grid_y = int((position[1] + 5) / self.grid_size)
        
        # 确保索引在范围内
        if 0 <= grid_x < 20 and 0 <= grid_y < 20:
            if self.visited_areas[grid_x, grid_y] == 0:
                reward_components['exploration'] = self.reward_weights['exploration']
                self.visited_areas[grid_x, grid_y] = 1
        
        # 9. 稳定性奖励 - 惩罚过度旋转或不稳定运动
        if abs(angular_velocity[2]) > 2.0:  # 只关注绕z轴的旋转
            reward_components['stability'] = -self.reward_weights['stability'] * abs(angular_velocity[2])
        
        # 10. 任务进度奖励 - 随着步数增加而增加奖励难度
        progress_factor = 1.0 + 0.5 * (step / max_steps)
        
        # 根据难度额外调整奖励
        difficulty_factor = 1.0 + difficulty
        
        # 更新上一步的状态
        self.prev_position = position
        self.prev_orientation = orientation
        self.prev_velocity = velocity
        
        # 计算总奖励
        total_reward = sum(reward_components.values()) * progress_factor * difficulty_factor
        
        return total_reward, reward_components
    
    def _get_min_mine_distance(self, vehicle, arena):
        """
        获取距离最近的雷区距离
        
        参数:
            vehicle: 车辆对象
            arena: 战斗场景
            
        返回:
            float: 最近雷区距离，如果没有雷区则返回无穷大
        """
        vehicle_pos = vehicle.get_position_and_orientation()[0][:2]  # 只取x, y坐标
        
        min_distance = float('inf')
        for mine_pos in arena.mine_positions:
            mine_pos_2d = mine_pos[:2]  # 只取x, y坐标
            distance = np.linalg.norm(np.array(vehicle_pos) - np.array(mine_pos_2d))
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取奖励计算器的统计信息
        
        返回:
            Dict[str, Any]: 包含各种统计指标的字典
        """
        return {
            'laser_hits': self.laser_hits,
            'total_efficiency': self.total_efficiency,
            'explored_percentage': np.sum(self.visited_areas) / (20 * 20) * 100,  # 探索百分比
            'reward_weights': self.reward_weights
        }
    
    def reset(self):
        """重置奖励计算器状态"""
        self.prev_position = None
        self.prev_orientation = None
        self.prev_velocity = None
        self.visited_areas = np.zeros((20, 20))
        self.laser_hits = 0
        self.total_efficiency = 0.0 