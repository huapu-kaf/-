"""
激光车辆强化学习 - 高级训练脚本
包含课程学习、增强奖励和复杂网络架构
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import pybullet as p
from collections import deque
import math
import random
import logging
import sys
import multiprocessing

# 检查GPU是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"可用的GPU数量: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # 设置默认设备为GPU
    torch.cuda.set_device(0)
    print(f"已设置默认GPU为: {torch.cuda.get_device_name(0)}")
else:
    print("警告: 未检测到GPU，将使用CPU进行训练，这会显著降低训练速度")

try:
    from laser_vehicle import LaserVehicle
    from combat_arena import CombatArena
    from reward_functions import RewardCalculator
except ImportError as e:
    print(f"导入自定义模块失败: {e}")
    print("请确保你在正确的目录下运行此脚本")
    exit(1)

# 新增: 可视化辅助函数
def setup_visualization(client_id):
    """
    设置PyBullet可视化参数和显示器
    
    参数:
        client_id: PyBullet客户端ID
    
    返回:
        display_ids: 显示元素的ID字典
    """
    # 配置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=6.0,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # 显示元素ID
    display_ids = {
        'episode_info': p.addUserDebugText(
            text="训练中...",
            textPosition=[0, -3, 1],
            textColorRGB=[1, 1, 1],
            textSize=1.5
        ),
        'reward_info': p.addUserDebugText(
            text="奖励: 0.0",
            textPosition=[0, -3, 0.7],
            textColorRGB=[0, 1, 0],
            textSize=1.2
        ),
        'action_info': p.addUserDebugText(
            text="动作: [0.0, 0.0]",
            textPosition=[0, -3, 0.4],
            textColorRGB=[0, 0.8, 1],
            textSize=1.2
        ),
        'training_stats': p.addUserDebugText(
            text="学习率: 0.0003 | 值损失: 0.0 | 策略损失: 0.0",
            textPosition=[0, -3, 0.1],
            textColorRGB=[1, 0.8, 0],
            textSize=1.0
        ),
        'controls_info': p.addUserDebugText(
            text="按ESC退出 | 空格键暂停/继续",
            textPosition=[0, -3, -0.2],
            textColorRGB=[0.8, 0.8, 0.8],
            textSize=1.0
        )
    }
    
    # 添加参数调节滑块
    display_ids['max_velocity_slider'] = p.addUserDebugParameter(
        paramName="最大速度",
        rangeMin=0.5,
        rangeMax=3.0,
        startValue=2.0
    )
    
    display_ids['reward_scale_slider'] = p.addUserDebugParameter(
        paramName="奖励缩放",
        rangeMin=0.1,
        rangeMax=2.0,
        startValue=1.0
    )
    
    return display_ids

def update_visualization(client, display_ids, info):
    """
    更新可视化信息
    
    参数:
        client: PyBullet客户端ID
        display_ids: 显示元素的ID字典
        info: 要显示的信息
    """
    # 更新各种显示信息
    if 'episode_info' in info and 'episode_info' in display_ids:
        episode_text = f"回合: {info['episode_info'].get('episode', 0)}, 步数: {info['episode_info'].get('steps', 0)}/{info['episode_info'].get('max_steps', 1000)}"
        p.addUserDebugText(
            episode_text,
            [0, 0, 2.0],
            textColorRGB=[1, 1, 1],
            textSize=1.5,
            replaceItemUniqueId=display_ids['episode_info'],
            physicsClientId=client
        )
    
    if 'reward_info' in info and 'reward_info' in display_ids:
        # 使用安全的get方法获取奖励信息，提供默认值
        reward_text = f"奖励: {info['reward_info'].get('total', 0.0):.2f}"
        
        # 添加详细奖励信息（如果存在）
        components = []
        for key in ['distance_reward', 'rotation_reward', 'collision_reward', 'laser_reward', 'goal_reward', 'time_penalty']:
            if key in info['reward_info']:
                components.append(f"{key.split('_')[0]}: {info['reward_info'][key]:.2f}")
        
        if components:
            reward_text += f" ({', '.join(components)})"
            
        p.addUserDebugText(
            reward_text,
            [0, 0, 1.8],
            textColorRGB=[0, 1, 0],
            textSize=1.2,
            replaceItemUniqueId=display_ids['reward_info'],
            physicsClientId=client
        )
    
    if 'status_info' in info and 'status_info' in display_ids:
        status_text = f"状态: {info['status_info'].get('status', 'Running')}, 难度: {info['status_info'].get('difficulty', 0.5):.1f}"
        p.addUserDebugText(
            status_text,
            [0, 0, 1.6],
            textColorRGB=[1, 1, 0],
            textSize=1.2,
            replaceItemUniqueId=display_ids['status_info'],
            physicsClientId=client
        )

class CustomCNN(BaseFeaturesExtractor):
    """
    自定义神经网络特征提取器，专为激光对抗任务设计
    处理160维观测空间(40维/时间步 * 4时间步)
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 应为160
        
        # 主网络结构 - 处理整个观测
        self.main_network = nn.Sequential(
            nn.Linear(n_input_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 为历史数据创建LSTM编码器
        self.history_encoder = nn.LSTM(
            input_size=40,  # 每个时间步的观测维度
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        
        # 处理LSTM输出的全连接层
        self.lstm_fc = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 使用主网络处理完整观测
        main_features = self.main_network(observations)
        
        # 确保注意力权重的维度正确
        attention_weights = self.attention(main_features)
        
        try:
            # 将160维观测重塑为(batch_size, 4, 40)以适应LSTM
            # 每个观测包含4个时间步，每个时间步40维
            history_obs = observations.reshape(batch_size, 4, 40)
            
            # 通过LSTM处理时间序列数据
            lstm_out, _ = self.history_encoder(history_obs)
            
            # 我们只关心最后一个时间步的输出
            lstm_features = lstm_out[:, -1, :]
            
            # 通过全连接层处理LSTM特征
            history_features = self.lstm_fc(lstm_features)
            
            # 使用注意力机制组合特征
            # 确保注意力权重的维度与特征匹配
            attention_weights = attention_weights.view(batch_size, 1)
            final_features = main_features * attention_weights + history_features * (1 - attention_weights)
            
            return final_features
            
        except Exception as e:
            # 如果重塑操作失败，打印错误信息并回退到只使用主网络特征
            print(f"LSTM处理失败: {e}, 观测形状: {observations.shape}")
            print(f"回退到只使用主网络特征")
            return main_features

class CurriculumCallback(BaseCallback):
    """
    课程学习回调，用于动态调整训练难度
    """
    def __init__(self, check_freq: int = 1000, 
                 initial_difficulty: float = 0.3,
                 difficulty_increment: float = 0.1,
                 success_threshold: float = 0.6):
        super(CurriculumCallback, self).__init__()
        
        self.check_freq = check_freq
        self.current_difficulty = initial_difficulty
        self.difficulty_increment = difficulty_increment
        self.success_threshold = success_threshold
        self.episode_rewards = []
        self.win_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        if len(self.episode_rewards) >= self.check_freq:
            success_rate = self.win_count / self.episode_count
            
            if success_rate >= self.success_threshold:
                self.current_difficulty = min(1.0, 
                    self.current_difficulty + self.difficulty_increment)
                print(f"\n提升难度至: {self.current_difficulty:.2f}")
            
            self.episode_rewards = []
            self.win_count = 0
            self.episode_count = 0
            
            # 更新环境难度
            self.training_env.env_method(
                "set_difficulty", 
                self.current_difficulty
            )
        
        return True

    def update_stats(self, reward: float, win: bool):
        self.episode_rewards.append(reward)
        self.episode_count += 1
        if win:
            self.win_count += 1

class AdvancedLaserVehicleEnv(gym.Env):
    """高级激光车环境"""
    
    def __init__(self, render_mode='none', difficulty=0.5, reward_config=None, observation_config=None, max_steps=1000):
        """
        初始化环境
        
        参数:
            render_mode: 渲染模式 ('none', 'direct', 'gui')
            difficulty: 难度级别 (0.0-1.0)
            reward_config: 奖励配置
            observation_config: 观测配置
            max_steps: 每局游戏最大步数
        """
        super().__init__()
        
        # 配置
        self.render_mode = render_mode
        self.difficulty = np.clip(difficulty, 0.1, 1.0)
        self.max_steps = max_steps
        
        # 激光照射相关状态
        self.laser_hit = False
        self.continuous_hit_start = None
        self.laser_line_id = None
        self.boundary_crossed = False
        
        # 游戏状态
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.laser_hits_count = 0
        # 坐标记录 - 先设置为None，等arena创建后再初始化
        self.start_position = None
        self.opponent_position = None
        self.mine_positions = []
        
        # 奖励配置 - 优化奖励权重
        self.reward_config = reward_config or {
            'base': {'weight': 1.0},  # 基础生存奖励
            'distance': {'weight': 3.0},  # 增加移动奖励权重
            'rotation': {'weight': 1.0},  # 增加旋转奖励权重
            'collision': {'weight': 15.0},  # 增加碰撞惩罚权重
            'laser': {'weight': 10.0},  # 大幅增加激光命中奖励
            'goal': {'weight': 15.0},  # 增加达成目标奖励
            'time': {'weight': 0.15}  # 减小时间惩罚
        }
        
        # 观测配置
        self.observation_config = observation_config or {
            'history_length': 4,
            'normalize': True,
            'lasers': 16,  # 修改为16，与实际使用的激光传感器数量匹配
            'include_velocity': True,
            'include_position': True,
        }
        
        # 内部状态
        self.steps = 0
        self.done = False
        self.episode_reward = 0.0
        
        # 奖励指标列表 - 在step和reset期间使用
        self.reward_metrics_keys = [
            'distance_reward', 
            'rotation_reward', 
            'collision_reward', 
            'laser_reward',
            'goal_reward', 
            'time_penalty',
            'total_reward'
        ]
        
        # 奖励指标数据
        self.episode_metrics = {key: 0.0 for key in self.reward_metrics_keys}
        
        # 遥测和日志
        self.enable_telemetry = False
        self.logger = self._setup_logger()
        self.debug = True
        
        # 动作和观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 观测空间尺寸:
        # - 激光距离: n_lasers (16)
        # - 速度: 3 (vx, vy, vz)
        # - 位置和方向: 6 (x, y, z, roll, pitch, yaw)
        # - 角速度: 3 (wx, wy, wz)
        # - 对手信息: 4
        # - 边界信息: 2
        # - 地雷信息: 2
        # - 激光信息: 2
        # - 额外特征: 2
        # 总计: n_lasers + 24 = 40维
        # 考虑历史长度4，最终维度为 40 * 4 = 160维
        n_inputs = (self.observation_config['lasers'] + 24) * self.observation_config['history_length']
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_inputs,), dtype=np.float32
        )
        
        # 连接PyBullet
        self.logger.info(f"正在连接PyBullet引擎，模式: {render_mode}")
        if render_mode == 'gui':
            self.client = p.connect(p.GUI)
            # 设置GUI相机
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=0,
                cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0]
            )
            # 添加车辆视角窗口
            self.car_view_window = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.001],
                                                     rgbaColor=[1, 1, 1, 0.5])
        else:
            self.client = p.connect(p.DIRECT)
        
        self.logger.info(f"PyBullet引擎连接成功，客户端ID: {self.client}")
        
        # 初始化物理环境
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(1.0/240.0, physicsClientId=self.client)
        
        # 创建战斗场景
        self.arena = CombatArena(render_mode=render_mode, debug=self.debug, client_id=self.client)
        
        # 现在初始化坐标记录 - 在创建arena后设置
        self.start_position = [-self.arena.arena_half_size + 0.5, -self.arena.arena_half_size + 0.5, 0.1]
        
        # 奖励计算器
        self.reward_calculator = RewardCalculator()
        
        # 可视化数据
        self.visualization_data = {
            'episode_info': {'episode': 0, 'steps': 0},
            'reward_info': {
                'total': 0.0,
                'survival': 0.0,
                'movement': 0.0,
                'target': 0.0,
                'laser': 0.0,
                'efficiency': 0.0,
                'collision': 0.0
            },
            'action_info': [0.0, 0.0],
            'speed_info': {'linear': 0.0, 'angular': 0.0},
            'training_stats': {
                'learning_rate': 0.0003,
                'value_loss': 0.0,
                'policy_loss': 0.0
            }
        }
        
        # 可视化显示元素
        self.display_ids = None
        if render_mode == 'gui':
            self.display_ids = setup_visualization(self.client)
    
    def set_difficulty(self, difficulty: float):
        """设置环境难度"""
        self.difficulty = np.clip(difficulty, 0.1, 1.0)
    
    def _reset_environment(self, env=None):
        """重置环境"""
        try:
            # 在这个方法中，我们不使用env参数，直接重置自己
            # 重置战斗场景
            self.arena.reset()
            
            # 创建主车辆
            start_pos = self._get_random_start_position()
            start_orientation = [0, 0, np.random.uniform(-np.pi, np.pi)]
            
            self.arena.spawn_vehicle(
                position=start_pos,
                orientation=start_orientation,
                mass=1.5,
                size=(0.2, 0.15, 0.1)
            )
            
            # 创建对手车辆 - 新增
            # 对手车辆在场地对角位置
            if start_pos[0] < 0:  # 如果主车在左侧，对手在右侧
                opponent_pos = [self.arena.arena_half_size - 0.5, self.arena.arena_half_size - 0.5, 0.1]
            else:  # 如果主车在右侧，对手在左侧
                opponent_pos = [-self.arena.arena_half_size + 0.5, -self.arena.arena_half_size + 0.5, 0.1]
                
            opponent_orientation = [0, 0, np.random.uniform(-np.pi, np.pi)]
            
            # 保存对手位置供其他方法使用
            self.opponent_position = opponent_pos
            
            # 生成对手车辆
            self.opponent_id = self.arena.spawn_vehicle(
                position=opponent_pos,
                orientation=opponent_orientation,
                mass=1.5,
                size=(0.2, 0.15, 0.1)
            )
            
            # 设置互为对手关系
            if hasattr(self.arena, 'vehicle') and self.arena.vehicle is not None:
                self.arena.vehicle.set_opponent(self.opponent_id)
            
            # 根据难度生成地雷
            mine_count = int(self.difficulty * 8)  # 难度越高，地雷越多
            self.mine_positions = []
            
            for _ in range(mine_count):
                # 在场地中央区域随机生成地雷
                mine_x = np.random.uniform(-2.0, 2.0)
                mine_y = np.random.uniform(-2.0, 2.0)
                mine_pos = [mine_x, mine_y, 0.05]
                self.mine_positions.append(mine_pos)
                
                # 在arena中添加地雷
                if hasattr(self.arena, 'spawn_mine'):
                    self.arena.spawn_mine(position=mine_pos)
            
            # 获取初始观测
            observation = self._get_observation()
            
            # 更新可视化信息
            if self.render_mode == 'gui' and self.display_ids:
                self.visualization_data['episode_info']['steps'] = 0
                update_visualization(
                    self.client,
                    self.display_ids,
                    self.visualization_data
                )
            
            return observation, {}
            
        except Exception as e:
            self.logger.error(f"重置环境时出错: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            # 返回零向量作为观测和空info字典
            if hasattr(self, 'observation_space') and self.observation_space is not None:
                dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            else:
                # 如果不知道观测空间，返回一个通用的零向量
                self.logger.warning("未知观测空间，返回默认观测值")
                dummy_obs = np.zeros(160)  # 使用我们知道的观测维度
            return dummy_obs, {}
    
    def reset(self, seed=None):
        """
        重置环境
        
        参数:
            seed: 随机种子
            
        返回:
            observation: 初始观测
            info: 环境信息
        """
        # 设置种子（如果提供）
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 重置环境状态
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.laser_hits_count = 0
        self.consecutive_laser_hits = 0
        self.last_laser_hit_time = 0
        self.reward_history = []
        
        # 重置环境并获取初始观测和信息
        observation, info = self._reset_environment()
        
        # 重置上一个动作
        self.previous_action = None
        
        return observation, info
    
    def _get_random_start_position(self):
        """获取随机起始位置"""
        # 随机选择左下角或右上角起始位置
        if np.random.rand() > 0.5:
            # 左下角
            return [-self.arena.arena_half_size + 0.5, -self.arena.arena_half_size + 0.5, 0.1]
        else:
            # 右上角
            return [self.arena.arena_half_size - 0.5, self.arena.arena_half_size - 0.5, 0.1]
    
    def step(self, action):
        """执行动作并转换环境"""
        self.steps += 1
        
        # 确保action是numpy数组
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)
        elif isinstance(action, np.ndarray):
            # 确保是正确的数据类型
            action = action.astype(np.float32)
        
        self.current_action = action.copy()
        
        # 应用动作到车辆，增加速度系数
        scaled_action = action * 3.0  # 增加速度
        self.arena.vehicle.apply_action(scaled_action)
        self.arena.step_simulation(1)
        
        # 更新车辆视角
        if self.render_mode == 'gui' and hasattr(self, 'car_view_window'):
            pos, ori = self.arena.vehicle.get_position_and_orientation()
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[pos[0], pos[1], pos[2] + 0.3],
                cameraTargetPosition=[pos[0] + math.cos(ori[2]), pos[1] + math.sin(ori[2]), pos[2]],
                cameraUpVector=[0, 0, 1]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0,
                nearVal=0.1, farVal=100.0
            )
            width, height = 320, 240
            img = p.getCameraImage(width, height, view_matrix, proj_matrix,
                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # 检查是否跨过分界线
        self._check_boundary_crossed()
        
        # 检查激光照射
        hit = self._check_laser_hit()
        
        # 获取观测
        observation = self._get_observation()
        
        # 计算奖励
        reward, reward_components = self._compute_reward(action, hit)
        self.episode_reward += reward
        
        # 保存当前动作用于下一步计算动作平滑度
        self.previous_action = action.copy()
        
        # 更新奖励组成
        self.episode_metrics = reward_components
        
        # 判断是否结束
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # 构建信息字典
        info = {
            'episode_reward': self.episode_reward,
            'current_step': self.steps,
            'reward_components': self.episode_metrics,
            'difficulty': self.difficulty,
            'laser_hit': self.laser_hit,
            'continuous_hit_time': 0 if self.continuous_hit_start is None else time.time() - self.continuous_hit_start
        }
        
        # 更新可视化数据
        self._update_visualization_data(action, reward_components)
        
        # 如果是可视化模式，更新显示
        if self.render_mode == 'gui' and self.display_ids:
            update_visualization(
                self.client,
                self.display_ids,
                self.visualization_data
            )
        
        return observation, reward, terminated, truncated, info
    
    def _update_visualization_data(self, action, reward_components):
        """更新可视化数据"""
        # 更新回合信息
        self.visualization_data['episode_info'] = {
            'steps': self.steps,
            'max_steps': self.max_steps
        }
        
        # 更新奖励信息
        self.visualization_data['reward_info'] = reward_components
        
        # 更新状态信息
        self.visualization_data['status_info'] = {
            'status': 'Running',
            'difficulty': self.difficulty
        }
    
    def _get_observation(self):
        """获取当前观测"""
        try:
            if self.arena.vehicle is None:
                self.logger.warning("车辆不存在，返回零观测")
                return np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            # 获取激光距离 (16维)
            laser_distances = self.arena.vehicle.get_laser_distances()  # 16维
            
            # 获取位置和方向 (6维)
            position, orientation = self.arena.vehicle.get_position_and_orientation()  # 3维 + 3维
            
            # 获取速度 (3维)
            velocity = self.arena.vehicle.get_velocity()  # 3维
            
            # 获取角速度 (3维)
            angular_velocity = self.arena.vehicle.get_angular_velocity()  # 3维
            
            # 计算到对手的相对信息 (4维)
            opponent_info = np.zeros(4)
            if hasattr(self, 'opponent_position') and self.opponent_position is not None:
                # 计算到对手的距离和方向
                dx = self.opponent_position[0] - position[0]
                dy = self.opponent_position[1] - position[1]
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.arctan2(dy, dx)
                
                # 计算对手相对于车辆前进方向的角度
                relative_angle = angle - orientation[2]
                while relative_angle > np.pi:
                    relative_angle -= 2 * np.pi
                while relative_angle < -np.pi:
                    relative_angle += 2 * np.pi
                
                opponent_info[0] = distance  # 到对手的距离
                opponent_info[1] = relative_angle  # 到对手的相对角度
                opponent_info[2] = np.cos(relative_angle)  # 相对角度的余弦
                opponent_info[3] = np.sin(relative_angle)  # 相对角度的正弦
            
            # 计算到分界线的信息 (2维)
            boundary_info = np.zeros(2)
            boundary_info[0] = abs(position[1])  # 到分界线的距离（y坐标的绝对值）
            boundary_info[1] = 1.0 if hasattr(self, 'boundary_crossed') and self.boundary_crossed else 0.0  # 是否已跨过分界线
            
            # 计算到最近地雷的信息 (2维)
            mine_info = np.zeros(2)
            if hasattr(self, 'mine_positions') and self.mine_positions:
                min_mine_distance = float('inf')
                nearest_mine_angle = 0
                
                for mine_pos in self.mine_positions:
                    dx_mine = mine_pos[0] - position[0]
                    dy_mine = mine_pos[1] - position[1]
                    dist_mine = np.sqrt(dx_mine*dx_mine + dy_mine*dy_mine)
                    
                    if dist_mine < min_mine_distance:
                        min_mine_distance = dist_mine
                        nearest_mine_angle = np.arctan2(dy_mine, dx_mine) - orientation[2]
                        while nearest_mine_angle > np.pi:
                            nearest_mine_angle -= 2 * np.pi
                        while nearest_mine_angle < -np.pi:
                            nearest_mine_angle += 2 * np.pi
                
                mine_info[0] = min_mine_distance  # 到最近地雷的距离
                mine_info[1] = nearest_mine_angle  # 最近地雷的相对角度
            
            # 添加激光照射相关信息 (2维)
            laser_info = np.zeros(2)
            laser_info[0] = 1.0 if hasattr(self, 'laser_hit') and self.laser_hit else 0.0  # 是否命中对手
            laser_info[1] = 0.0 if not hasattr(self, 'continuous_hit_start') or self.continuous_hit_start is None else min(time.time() - self.continuous_hit_start, 10.0)  # 连续命中时间，限制最大值为10秒
            
            # 组合当前观测 (总共40维)
            current_observation = np.concatenate([
                laser_distances,   # 16维
                np.array(position),   # 3维
                np.array(orientation),   # 3维
                np.array(velocity),   # 3维
                np.array(angular_velocity),   # 3维
                opponent_info,   # 4维
                boundary_info,   # 2维
                mine_info,   # 2维
                laser_info,   # 2维
                np.zeros(2)   # 额外2维，使总维度达到40
            ]).astype(np.float32)
            
            # 打印调试信息
            if self.debug:
                self.logger.info(f"当前观测维度: {current_observation.shape[0]}")
                self.logger.info(f"激光距离: {laser_distances.shape[0]}维")
                self.logger.info(f"位置: {len(position)}维, 方向: {len(orientation)}维")
                self.logger.info(f"速度: {len(velocity)}维, 角速度: {len(angular_velocity)}维")
                self.logger.info(f"对手信息: {opponent_info.shape[0]}维")
                self.logger.info(f"边界信息: {boundary_info.shape[0]}维")
                self.logger.info(f"地雷信息: {mine_info.shape[0]}维")
                self.logger.info(f"激光信息: {laser_info.shape[0]}维")
            
            # 如果没有历史观测列表，创建一个
            if not hasattr(self, '_observation_history'):
                self._observation_history = deque(maxlen=self.observation_config['history_length'])
                # 用当前观测初始化历史
                for _ in range(self.observation_config['history_length']):
                    self._observation_history.append(current_observation.copy())
            
            # 更新历史观测
            self._observation_history.append(current_observation.copy())
            
            # 组合所有历史观测 (40维 * 4 = 160维)
            observation = np.concatenate(list(self._observation_history))
            
            # 打印最终观测维度
            if self.debug:
                self.logger.info(f"最终观测维度: {observation.shape[0]}")
            
            return observation
            
        except Exception as e:
            self.logger.error(f"获取观测时出错: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _compute_reward(self, action, info):
        """
        计算奖励值
        
        参数:
            action: 当前动作
            info: 附加信息
            
        返回:
            tuple: (总奖励值, 奖励组件字典)
        """
        try:
            if self.arena.vehicle is None:
                self.logger.warning("车辆不存在，返回零奖励")
                return 0.0, {key: 0.0 for key in self.reward_metrics_keys}
                
            # 基本移动和存活奖励
            reward_components = {
                'survival_reward': 0.01,        # 存活基础奖励
                'distance_reward': 0.0,         # 接近对手的奖励
                'rotation_reward': 0.0,         # 朝向对手的奖励
                'waypoint_reward': 0.0,         # 经过分界点的奖励
                'laser_reward': 0.0,            # 激光照射奖励
                'goal_reward': 0.0,             # 任务完成奖励
                'mine_penalty': 0.0,            # 靠近或触碰地雷的惩罚
                'collision_reward': 0.0,        # 碰撞惩罚
                'out_of_bounds': 0.0,           # 离开边界惩罚
                'smooth_reward': 0.0,           # 平滑动作奖励
                'exploration_reward': 0.0,      # 探索奖励
                'energy_conservation': 0.0,     # 能量节约奖励
                'time_penalty': -0.005,         # 时间惩罚
                'total_reward': 0.0             # 总奖励
            }
            
            # 获取当前状态信息
            position, orientation = self.arena.vehicle.get_position_and_orientation()
            velocity = self.arena.vehicle.get_velocity()
            speed = np.linalg.norm(velocity)
            
            # 检查info是否为字典类型，并获取laser_hit值
            if isinstance(info, dict):
                laser_hit = info.get('laser_hit', False)
            else:
                laser_hit = False
            
            # 保存当前位置以供其他地方使用
            self.current_position = position
            
            # 首先检查是否出界
            if abs(position[0]) > self.arena.arena_half_size or abs(position[1]) > self.arena.arena_half_size:
                reward_components['out_of_bounds'] = -5.0
                self.logger.info(f"车辆出界，惩罚: {reward_components['out_of_bounds']}")
            
            # 检查是否触碰地雷
            if hasattr(self.arena, 'mine_positions') and self.arena.mine_positions:
                for mine_pos in self.arena.mine_positions:
                    dx = position[0] - mine_pos[0]
                    dy = position[1] - mine_pos[1]
                    distance_to_mine = np.sqrt(dx*dx + dy*dy)
                    
                    # 如果距离小于地雷半径+车辆半径，认为触碰到地雷
                    if distance_to_mine < (0.075 + 0.15):  # 地雷半径(7.5cm) + 车辆宽度的一半(15cm)
                        reward_components['mine_penalty'] = -15.0  # 大幅增加地雷惩罚
                        self.logger.info(f"车辆触碰地雷！惩罚: {reward_components['mine_penalty']}")
                        break
                    # 如果接近地雷但未触碰，给予小的警告惩罚
                    elif distance_to_mine < (0.075 + 0.3):  # 在30cm范围内接近地雷
                        close_penalty = -2.0 * (1.0 - (distance_to_mine - 0.075) / 0.3)  # 越接近惩罚越大
                        reward_components['mine_penalty'] = min(reward_components['mine_penalty'], close_penalty)
                    else:
                        # 成功避开地雷（处于0.3~0.7m的合理距离），给予小奖励
                        if distance_to_mine < 0.7 and distance_to_mine > 0.3:
                            avoid_reward = 0.2 * ((distance_to_mine - 0.3) / 0.4)  # 奖励随距离增加而增加
                            reward_components['mine_penalty'] += avoid_reward
            
            # 检查碰撞（与墙壁或其他物体）
            collision = self.arena.vehicle.check_collision()
            if collision:
                reward_components['collision_reward'] = -5.0  # 增大碰撞惩罚
                self.logger.info(f"车辆碰撞，惩罚: {reward_components['collision_reward']}")
            
            # 计算到对手的距离和方向（如果对手存在）
            if hasattr(self, 'opponent_position') and self.opponent_position is not None:
                dx = self.opponent_position[0] - position[0]
                dy = self.opponent_position[1] - position[1]
                distance = np.sqrt(dx*dx + dy*dy)
                angle = np.arctan2(dy, dx)
                
                # 计算对手相对于车辆前进方向的角度
                relative_angle = angle - orientation[2]
                while relative_angle > np.pi:
                    relative_angle -= 2 * np.pi
                while relative_angle < -np.pi:
                    relative_angle += 2 * np.pi
                
                # 距离奖励：越接近对手奖励越高，但需要跨过分界线才生效
                if hasattr(self, 'boundary_crossed') and self.boundary_crossed:
                    # 理想距离是1米左右，既不太近也不太远
                    ideal_distance = 1.0
                    if distance < ideal_distance:
                        # 距离太近时，奖励随着接近而减少
                        proximity_factor = 1.0 - (ideal_distance - distance) / ideal_distance
                    else:
                        # 距离太远时，奖励随着距离增加而减少
                        proximity_factor = 1.0 - min(1.0, (distance - ideal_distance) / 2.0)
                    
                    reward_components['distance_reward'] = 0.8 * proximity_factor  # 增加奖励以鼓励保持理想距离
                
                # 方向奖励：车辆方向与对手方向对齐时奖励高，同样需要跨过分界线
                if hasattr(self, 'boundary_crossed') and self.boundary_crossed:
                    alignment = np.cos(relative_angle)  # 1表示完全对齐，0表示垂直，-1表示相反
                    reward_components['rotation_reward'] = 0.5 * max(0, alignment)  # 只有当正对对手时才给奖励
                
                # 如果完成了一圈（回到起点附近），给予额外奖励
                start_pos = self.start_positions[0] if hasattr(self, 'start_positions') else [-self.arena.arena_half_size + 0.5, -self.arena.arena_half_size + 0.5, 0.1]
                dx_start = position[0] - start_pos[0]
                dy_start = position[1] - start_pos[1]
                distance_to_start = np.sqrt(dx_start*dx_start + dy_start*dy_start)
                
                if distance_to_start < 0.5 and self.steps > 100:  # 确保已经离开过起点
                    reward_components['goal_reward'] += 3.0  # 增加鼓励回到起点的奖励
            
            # 跨过边界奖励
            if hasattr(self, 'boundary_crossed') and self.boundary_crossed:
                reward_components['waypoint_reward'] = 0.3  # 增加边界跨越奖励
            
            # 激光照射奖励 - 大幅增强以鼓励照射行为
            if laser_hit:
                # 基础照射奖励
                reward_components['laser_reward'] = 2.0  # 增加基础激光命中奖励
                
                # 检查是否持续照射超过2秒
                if hasattr(self, 'continuous_hit_start') and self.continuous_hit_start is not None:
                    hit_duration = time.time() - self.continuous_hit_start
                    # 随着持续照射时间增加，提供渐进增加的奖励
                    reward_components['laser_reward'] += min(5.0, hit_duration * 2.0)  # 随时间增加更快，最多5.0
                    
                    if hit_duration >= 2.0:  # 2秒连续照射
                        reward_components['goal_reward'] = 50.0  # 大幅增加胜利奖励
                        
                        # 如果胜利，打印信息
                        if self.render_mode == 'gui':
                            print("胜利！连续照射对手超过2秒")
            
            # 激光避障奖励：通过激光传感器安全地探测和避开障碍物
            laser_distances = self.arena.vehicle.get_laser_distances()
            # 修复NumPy数组的布尔检查
            min_laser = 999  # 默认值
            if laser_distances is not None and isinstance(laser_distances, (list, np.ndarray)) and len(laser_distances) > 0:
                min_laser = np.min(laser_distances)
            
            if min_laser < 0.3:  # 如果太接近障碍物
                obstacle_penalty = -0.5 * (0.3 - min_laser) / 0.3  # 更强的惩罚函数
                reward_components['laser_reward'] += obstacle_penalty
            elif laser_distances is not None and isinstance(laser_distances, (list, np.ndarray)) and len(laser_distances) > 0 and np.min(laser_distances) > 0.3:  # 所有激光都保持安全距离
                reward_components['laser_reward'] += 0.1  # 奖励安全导航
            
            # 速度奖励：鼓励车辆保持适当速度
            ideal_speed = 1.5  # 理想速度（米/秒）
            speed_diff = abs(speed - ideal_speed)
            if speed_diff < 0.5:
                reward_components['smooth_reward'] += 0.2 * (1.0 - speed_diff / 0.5)  # 增加速度接近理想值的奖励
            else:
                # 速度过高或过低的惩罚
                reward_components['smooth_reward'] -= 0.1 * min(1.0, (speed_diff - 0.5) / 1.0)
            
            # 平滑动作奖励：连续动作之间的差异越小越好
            if hasattr(self, 'previous_action') and self.previous_action is not None:
                action_diff = np.sum(np.abs(action - self.previous_action))
                smoothness = np.exp(-action_diff)  # 1表示完全平滑，接近0表示不平滑
                reward_components['smooth_reward'] += 0.2 * smoothness
            
            # 能量节约奖励：鼓励使用较小的动作值（节约能量）
            energy = np.sum(np.square(action))
            energy_factor = np.exp(-energy)  # 1表示零能量，接近0表示高能量
            reward_components['energy_conservation'] = 0.1 * energy_factor
            
            # 探索奖励：基于访问新区域
            if hasattr(self, 'visited_positions'):
                # 计算当前位置的网格坐标
                grid_x = int((position[0] + self.arena.arena_half_size) / 0.2)  # 20cm网格
                grid_y = int((position[1] + self.arena.arena_half_size) / 0.2)
                grid_pos = (grid_x, grid_y)
                
                # 如果是新访问的位置，给予奖励
                if grid_pos not in self.visited_positions:
                    self.visited_positions.add(grid_pos)
                    reward_components['exploration_reward'] = 0.1
            else:
                # 首次初始化
                self.visited_positions = set()
                grid_x = int((position[0] + self.arena.arena_half_size) / 0.2)
                grid_y = int((position[1] + self.arena.arena_half_size) / 0.2)
                self.visited_positions.add((grid_x, grid_y))
            
            # 保存当前动作以供下一步使用
            self.previous_action = action
            
            # 计算总奖励
            total_reward = sum(reward_components.values())
            reward_components['total_reward'] = total_reward
            
            # 记录每100步的奖励值
            if self.steps % 100 == 0:
                self.logger.debug(f"步数: {self.steps}, 奖励: {total_reward}")
            
            return total_reward, reward_components
            
        except Exception as e:
            self.logger.error(f"计算奖励时出错: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return 0.0, {key: 0.0 for key in self.reward_metrics_keys}
    
    def _is_terminated(self):
        """检查是否终止（例如碰撞、地雷或胜利）"""
        # 检查碰撞
        if self.arena.vehicle.check_collision():
            return True
        
        # 检查是否触碰地雷
        position, _ = self.arena.vehicle.get_position_and_orientation()
        if hasattr(self.arena, 'mine_objects') and self.arena.mine_objects:
            for mine_pos in self.arena.mine_positions:
                dx = position[0] - mine_pos[0]
                dy = position[1] - mine_pos[1]
                distance_to_mine = np.sqrt(dx*dx + dy*dy)
                
                # 如果距离小于地雷半径+车辆半径，认为触碰到地雷
                if distance_to_mine < (0.075 + 0.15):  # 地雷半径(7.5cm) + 车辆宽度的一半(15cm)
                    self.logger.info(f"车辆触碰地雷，终止回合！")
                    return True
        
        # 检查是否连续照射对手2秒以上（胜利条件）
        if self.continuous_hit_start is not None:
            hit_duration = time.time() - self.continuous_hit_start
            if hit_duration >= 2.0:
                self.logger.info(f"连续照射对手{hit_duration:.2f}秒，达成胜利条件！")
                return True
        
        # 检查是否出界
        if (abs(position[0]) > 3.5 or abs(position[1]) > 3.5):
            self.logger.info(f"车辆出界，终止回合！")
            return True
        
        return False
    
    def _is_truncated(self):
        """检查是否截断（例如达到最大步数）"""
        return self.steps >= self.max_steps
    
    def render(self):
        """渲染环境"""
        pass  # PyBullet已经在内部渲染
    
    def close(self):
        """关闭环境"""
        self.arena.close()

    def _setup_logger(self):
        """设置日志记录器"""
        import logging
        
        # 创建logger
        logger = logging.getLogger(f"LaserVehicleEnv")
        logger.setLevel(logging.INFO)
        
        # 检查是否已有处理器，避免重复
        if not logger.handlers:
            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(ch)
            
            # 文件处理器
            try:
                # 确保日志目录存在
                log_dir = './logs'
                os.makedirs(log_dir, exist_ok=True)
                
                # 添加文件处理器
                fh = logging.FileHandler(f'{log_dir}/env.log')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                logger.warning(f"无法创建文件日志处理器: {e}")
        
        return logger
        
    def _log_telemetry(self, event_type, data=None):
        """记录遥测数据"""
        if not self.enable_telemetry:
            return
            
        if data is None:
            data = {}
            
        # 添加基本信息
        telemetry_data = {
            "event_type": event_type,
            "timestamp": time.time(),
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "metrics": self.episode_metrics,
            **data
        }
        
        # 记录遥测
        self.logger.debug(f"遥测: {telemetry_data}")
        
        # 这里可以添加发送到遥测服务器的代码
        # 例如使用wandb等

    def _check_boundary_crossed(self):
        """检查是否跨过中心分界线"""
        if not hasattr(self, 'boundary_crossed'):
            self.boundary_crossed = False
            
        # 获取车辆位置
        position, _ = self.arena.vehicle.get_position_and_orientation()
        
        # 检查是否跨过y=0的分界线
        if position[1] > 0 and not self.boundary_crossed:
            self.boundary_crossed = True
            if self.render_mode == 'gui':
                print("跨过分界线，可以进行激光照射！")
    
    def _check_laser_hit(self):
        """检查激光是否命中对手车辆"""
        # 获取我方车辆位置和朝向
        position, orientation = self.arena.vehicle.get_position_and_orientation()
        
        # 如果还没有跨过分界线，不能射击
        if not hasattr(self, 'boundary_crossed') or not self.boundary_crossed:
            self.laser_hit = False
            self.continuous_hit_start = None
            # 删除上一帧的激光线
            if self.render_mode == 'gui' and hasattr(self, 'laser_line_id') and self.laser_line_id is not None:
                p.removeUserDebugItem(self.laser_line_id)
                self.laser_line_id = None
            return False
            
        # 对手车辆不存在，返回False
        if not hasattr(self, 'opponent_id') or not hasattr(self, 'opponent_position') or self.opponent_position is None:
            return False
            
        # 计算激光方向向量（车辆前进方向）
        forward_x = math.cos(orientation[2])
        forward_y = math.sin(orientation[2])
        
        # 激光起始点（从车辆前部射出）
        laser_start = [
            position[0] + forward_x * 0.15,  # 车前部
            position[1] + forward_y * 0.15,
            position[2] + 0.05  # 略高于车身
        ]
        
        # 激光最大射程
        max_range = 10.0
        laser_end = [
            laser_start[0] + forward_x * max_range,
            laser_start[1] + forward_y * max_range,
            laser_start[2]
        ]
        
        # 执行射线检测
        results = p.rayTest(laser_start, laser_end)
        
        # 射线命中结果
        hit = False
        hit_position = None
        hit_object_id = -1
        
        # 解析结果
        if results and results[0][0] != -1:  # 有物体被击中
            hit_object_id = results[0][0]
            hit_position = results[0][3]  # 命中位置
            
            # 检查是否击中对手车辆
            if hit_object_id == self.opponent_id:
                hit = True
                # 如果是第一次命中，记录时间
                if not self.laser_hit:
                    self.laser_hit = True
                    self.continuous_hit_start = time.time()
                    if self.render_mode == 'gui':
                        print("激光命中对手车辆！")
            else:
                # 未命中对手，重置状态
                self.laser_hit = False
                self.continuous_hit_start = None
        else:
            # 没有命中任何物体，重置状态
            self.laser_hit = False
            self.continuous_hit_start = None
            
        # 在GUI模式下可视化激光线
        if self.render_mode == 'gui':
            # 删除上一帧的激光线
            if hasattr(self, 'laser_line_id') and self.laser_line_id is not None:
                p.removeUserDebugItem(self.laser_line_id)
                
            # 激光线颜色：命中为红色，未命中为绿色
            color = [1, 0, 0] if hit else [0, 1, 0]
            
            # 绘制激光线到命中点或最大距离
            end_point = hit_position if hit_position else laser_end
            self.laser_line_id = p.addUserDebugLine(
                laser_start,
                end_point,
                color,
                1.0,  # 线宽
                0.1   # 持续时间
            )
            
        return hit

# 新增: 可视化回调类
class VisualizationCallback(BaseCallback):
    """用于更新训练过程中的可视化信息"""
    
    def __init__(self, verbose=0):
        super(VisualizationCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # 获取训练统计信息
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            stats = self.model.logger.name_to_value
            
            # 更新环境中的训练统计信息
            # 只在GUI模式下更新可视化信息
            try:
                # 对于DummyVecEnv，可以直接访问envs
                if hasattr(self.training_env, 'envs'):
                    for env in self.training_env.envs:
                        if hasattr(env, 'visualization_data') and hasattr(env, 'render_mode') and env.render_mode == 'gui':
                            self._update_env_stats(env, stats)
                # 对于SubprocVecEnv，需要使用env_method
                elif hasattr(self.training_env, 'env_method'):
                    # 这里我们不能直接更新，因为SubprocVecEnv中的环境在不同进程中
                    # 如果需要，可以通过env_method传递信息
                    pass
            except Exception as e:
                # 忽略可视化错误，不影响训练
                if self.verbose > 0:
                    print(f"可视化更新错误: {e}")
        
        return True
        
    def _update_env_stats(self, env, stats):
        """更新单个环境的统计信息"""
        if 'train/learning_rate' in stats:
            env.visualization_data['training_stats']['learning_rate'] = stats['train/learning_rate']
        if 'train/value_loss' in stats:
            env.visualization_data['training_stats']['value_loss'] = stats['train/value_loss']
        if 'train/policy_gradient_loss' in stats:
            env.visualization_data['training_stats']['policy_loss'] = stats['train/policy_gradient_loss']

def make_env(rank, seed=0, env_args=None, verbose=0, log_dir='./logs'):
    """
    创建单个环境实例的辅助函数
    用于VecEnv的并行环境创建
    
    参数:
        rank: 环境编号
        seed: 随机种子
        env_args: 环境参数字典
        verbose: 详细程度
        log_dir: 日志目录
    
    返回:
        env_fn: 创建环境的函数
    """
    if env_args is None:
        env_args = {}
        
    def _init():
        try:
            # 创建环境
            env = AdvancedLaserVehicleEnv(**env_args)
            
            # 设置唯一的种子
            env_seed = seed + rank * 1000 if seed is not None else None
            # 不再使用env.seed方法，而是在reset时传递种子
            if env_seed is not None:
                # 调用reset时传递种子，而不是使用已弃用的seed方法
                obs, _ = env.reset(seed=env_seed)
            
            # 包装监控器
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                env = Monitor(env, os.path.join(log_dir, f'env_{rank}'), allow_early_resets=True)
            
            return env
        except Exception as e:
            print(f"创建环境 {rank} 时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个简单的空环境，避免整个训练失败
            return gymnasium.Env()
            
    return _init

def train_advanced_rl(config):
    """
    高级强化学习训练函数
    
    参数:
        config: 训练配置字典
        
    返回:
        训练好的模型
    """
    print(f"训练配置: {config}")
    
    # 确保日志目录存在
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['log_dir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['log_dir'], 'tensorboard'), exist_ok=True)
    
    # 保存配置
    serializable_config = {}
    for k, v in config.items():
        if k == 'policy_kwargs':
            # 处理策略参数
            serializable_kwargs = {}
            for kw_key, kw_val in v.items():
                if kw_key == 'features_extractor_class':
                    serializable_kwargs[kw_key] = v[kw_key].__name__  # 只保存类名
                elif kw_key == 'activation_fn' and v[kw_key] is not None:
                    serializable_kwargs[kw_key] = v[kw_key].__name__ if hasattr(v[kw_key], '__name__') else str(v[kw_key])
                else:
                    serializable_kwargs[kw_key] = kw_val
            serializable_config[k] = serializable_kwargs
        else:
            # 其他普通参数
            if callable(v):
                serializable_config[k] = v.__name__ if hasattr(v, '__name__') else str(v)
            else:
                serializable_config[k] = v
    
    try:
        with open(os.path.join(config['log_dir'], 'config.json'), 'w') as f:
            json.dump(serializable_config, f, indent=2)
    except TypeError as e:
        print(f"警告：无法序列化配置，将跳过保存：{e}")
    
    # 设置随机种子
    set_random_seed(config['seed'])
    
    # 创建环境
    try:
        # 检测可用CPU核心数，用于设置合理的并行环境数
        available_cores = multiprocessing.cpu_count()
        print(f"可用CPU核心数: {available_cores}")
        
        # 调整n_envs，不超过可用核心数-1（保留一个核心给主进程）
        if config['n_envs'] > available_cores - 1 and config['n_envs'] > 1:
            recommended_envs = max(1, available_cores - 1)
            print(f"警告: 请求的并行环境数 ({config['n_envs']}) 超过可用CPU核心数-1 ({available_cores-1})")
            print(f"建议将n_envs设置为 {recommended_envs}")
            if not config['force_n_envs']:  # 如果没有强制使用指定数量
                config['n_envs'] = recommended_envs
                print(f"已自动调整n_envs为 {config['n_envs']}")
            else:
                print("由于force_n_envs=True，保持请求的环境数")
        
        if config['render_mode'] == 'gui':
            # GUI模式下强制使用单环境
            print(f"GUI模式训练，使用单环境")
            env = AdvancedLaserVehicleEnv(render_mode='gui', difficulty=config.get('difficulty', 0.5))
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
        elif config['n_envs'] > 1:
            # 多环境训练，全部使用direct模式
            print(f"创建{config['n_envs']}个并行环境...")
            
            # 基础环境参数
            base_env_args = {
                'render_mode': 'direct',
                'difficulty': config.get('difficulty', 0.5),
                'max_steps': config.get('max_steps', 1000)
            }
            
            # 创建环境构建函数
            env_fns = []
            for i in range(config['n_envs']):
                # 为每个环境创建独立的参数副本
                env_args = base_env_args.copy()
                # 每个环境使用不同的种子
                env_seed = config['seed'] + i * 1000 if config['seed'] is not None else i * 1000
                env_fns.append(make_env(i, env_seed, env_args, log_dir=os.path.join(config['log_dir'], 'logs')))
            
            # 使用子进程向量环境
            try:
                env = SubprocVecEnv(env_fns)
                print("成功创建SubprocVecEnv")
            except Exception as e:
                print(f"创建SubprocVecEnv失败: {e}，回退到DummyVecEnv")
                # 如果子进程环境创建失败，回退到串行环境
                env = DummyVecEnv(env_fns)
            
            # 添加监控
            env = VecMonitor(env)
        else:
            # 单环境训练
            print(f"创建单环境，渲染模式: {config['render_mode']}")
            env = AdvancedLaserVehicleEnv(
                render_mode=config['render_mode'],
                difficulty=config.get('difficulty', 0.5),
                max_steps=config.get('max_steps', 1000)
            )
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            
        print("环境创建成功")
    except Exception as e:
        print(f"创建环境时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 创建保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('save_freq', 10000) // config['n_envs'],  # 每N步保存一次模型
        save_path=os.path.join(config['log_dir'], 'checkpoints'),
        name_prefix="model"
    )
    
    curriculum_callback = CurriculumCallback(
        check_freq=config.get('save_freq', 10000) // config['n_envs'] // 2,  # 更频繁地检查课程学习状态
        initial_difficulty=0.3,  # 提高初始难度
        difficulty_increment=0.1,  # 每次成功后增加的难度
        success_threshold=0.6  # 认为成功的平均奖励阈值
    )
    
    # 新增: 可视化回调
    visualization_callback = VisualizationCallback()
    
    # 构建模型
    try:
        print("创建PPO模型...")
        
        # 提取PPO特定参数
        ppo_kwargs = {
            'policy': config['policy_type'],
            'env': env,
            'verbose': 1,
            'tensorboard_log': os.path.join(config['log_dir'], 'tensorboard'),
            'learning_rate': config['learning_rate'],
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'n_epochs': config['n_epochs'],
            'gamma': config['gamma'],
            'clip_range': config['clip_range'],
            'policy_kwargs': config.get('policy_kwargs', None)
        }
        
        # 添加可选的高级参数
        for key in ['gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm', 'clip_range_vf']:
            if key in config:
                ppo_kwargs[key] = config[key]
                
        model = PPO(**ppo_kwargs)
        print("模型创建成功")
    except Exception as e:
        print(f"创建模型时出错: {e}")
        try:
            env.close()
        except:
            pass
        return None
    
    # 开始训练
    start_time = time.time()
    print(f"开始高级训练，总步数: {config['total_timesteps']}")
    
    try:
        # 在GUI模式下，添加额外的处理
        if config['render_mode'] == 'gui':
            print("GUI模式训练中，请注意观察PyBullet窗口...")
            # 确保PyBullet窗口已经初始化
            time.sleep(1.0)
        
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=[checkpoint_callback, curriculum_callback, visualization_callback],
            progress_bar=True  # 显示进度条
        )
        
        # 保存最终模型
        final_model_path = os.path.join(config['log_dir'], 'final_model')
        model.save(final_model_path)
        print(f"模型已保存至: {final_model_path}")
    except KeyboardInterrupt:
        print("用户中断训练")
        # 保存中断时的模型
        interrupted_model_path = os.path.join(config['log_dir'], 'interrupted_model')
        model.save(interrupted_model_path)
        print(f"已保存中断时的模型至: {interrupted_model_path}")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
    finally:
        # 确保正确关闭环境
        try:
            print("关闭环境...")
            env.close()
            print("环境已关闭")
        except Exception as e:
            print(f"关闭环境时出错: {e}")
    
    end_time = time.time()
    print(f"训练完成！用时: {end_time - start_time:.2f} 秒")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="激光车辆强化学习高级训练脚本")
    parser.add_argument('--model_type', type=str, default='PPO',
                      help='强化学习算法类型 (目前仅支持PPO)')
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                      help='总训练步数')
    parser.add_argument('--n_envs', type=int, default=6,
                      help='并行环境数量')
    parser.add_argument('--render_mode', type=str, default='direct',
                      help='渲染模式 (direct/gui)')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--log_dir', type=str, default='./advanced_logs',
                      help='日志目录')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='学习率')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='批次大小')
    parser.add_argument('--n_steps', type=int, default=4096,
                      help='每次收集的步数')
    parser.add_argument('--n_epochs', type=int, default=10,
                      help='每批数据的训练轮数')
    parser.add_argument('--gamma', type=float, default=0.995,
                      help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.98,
                      help='GAE lambda参数')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                      help='熵系数')
    parser.add_argument('--continue_training', action='store_true',
                      help='是否从上次保存的模型继续训练')
    parser.add_argument('--model_path', type=str, default=None,
                      help='继续训练时加载的模型路径')
    parser.add_argument('--force_n_envs', action='store_true',
                      help='强制使用指定的并行环境数量，即使超出CPU核心数')
    parser.add_argument('--difficulty', type=float, default=0.7,
                      help='难度级别 (0.1-1.0)，越高越难')
    parser.add_argument('--max_steps', type=int, default=1500,
                      help='每个回合的最大步数')
    parser.add_argument('--save_freq', type=int, default=50000,
                      help='保存模型的频率（步数）')
    
    args = parser.parse_args()
    
    # GUI模式下强制单环境
    if args.render_mode == 'gui':
        args.n_envs = 1
        print("GUI模式下只能使用单环境，已自动调整")
    
    # 构建配置字典
    config = {
        'model_type': args.model_type,
        'policy_type': 'MlpPolicy',
        'total_timesteps': args.total_timesteps,
        'n_envs': args.n_envs,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'learning_rate': args.learning_rate,
        'clip_range': args.clip_range if hasattr(args, 'clip_range') else 0.2,
        'clip_range_vf': 0.3,  # 价值函数裁剪范围
        'ent_coef': args.ent_coef,
        'vf_coef': 0.5,  # 价值函数系数
        'max_grad_norm': 0.5,  # 梯度裁剪
        'seed': args.seed,
        'log_dir': args.log_dir,
        'save_freq': args.save_freq,  # 保存模型的频率
        'render_mode': args.render_mode,
        'force_n_envs': args.force_n_envs,  # 是否强制使用指定的并行环境数
        'difficulty': args.difficulty,  # 难度级别
        'max_steps': args.max_steps,  # 每个回合的最大步数
        'policy_kwargs': {
            'net_arch': [
                {'pi': [512, 384, 256], 'vf': [512, 384, 256]}
            ],
            'features_extractor_class': CustomCNN,
            'features_extractor_kwargs': {'features_dim': 512},  # 保持特征维度较大
            'activation_fn': nn.ReLU    # 使用ReLU激活函数
        }
    }
    
    # 检查是否继续训练
    if args.continue_training and args.model_path:
        print(f"继续从模型 {args.model_path} 训练")
        # 创建环境
        env = AdvancedLaserVehicleEnv(render_mode=args.render_mode)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # 加载模型
        try:
            model = PPO.load(args.model_path, env=env)
            print("模型加载成功，继续训练...")
            
            # 创建回调
            checkpoint_callback = CheckpointCallback(
                save_freq=config.get('save_freq', 10000) // config['n_envs'],
                save_path=os.path.join(config['log_dir'], 'checkpoints'),
                name_prefix='model_continued'
            )
            
            curriculum_callback = CurriculumCallback(
                check_freq=config.get('save_freq', 10000) // config['n_envs'] // 2,
                initial_difficulty=0.5,  # 继续训练时从更高难度开始
                difficulty_increment=0.1,
                success_threshold=0.7
            )
            
            visualization_callback = VisualizationCallback()
            
            # 继续训练
            start_time = time.time()
            model.learn(
                total_timesteps=config['total_timesteps'],
                callback=[checkpoint_callback, curriculum_callback, visualization_callback],
                reset_num_timesteps=False  # 不重置计数器
            )
            
            # 保存最终模型
            final_model_path = os.path.join(config['log_dir'], 'final_model_continued')
            model.save(final_model_path)
            print(f"继续训练后的模型已保存至: {final_model_path}")
            
            end_time = time.time()
            print(f"继续训练完成！用时: {end_time - start_time:.2f} 秒")
        except Exception as e:
            print(f"继续训练时出错: {e}")
    else:
        # 从头开始训练
        # 确保日志目录存在
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['log_dir'], 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config['log_dir'], 'tensorboard'), exist_ok=True)
        
        # 记录训练配置
        try:
            with open(os.path.join(config['log_dir'], 'training_config.json'), 'w') as f:
                # 将不可序列化的对象转换为字符串
                serializable_config = {}
                for k, v in config.items():
                    if k == 'policy_kwargs':
                        serializable_kwargs = {}
                        for kw_key, kw_val in v.items():
                            if kw_key == 'features_extractor_class':
                                serializable_kwargs[kw_key] = kw_val.__name__ if hasattr(kw_val, '__name__') else str(kw_val)
                            elif kw_key == 'activation_fn' and kw_val is not None:
                                serializable_kwargs[kw_key] = kw_val.__name__ if hasattr(kw_val, '__name__') else str(kw_val)
                            else:
                                try:
                                    json.dumps(kw_val)  # 尝试序列化
                                    serializable_kwargs[kw_key] = kw_val
                                except:
                                    serializable_kwargs[kw_key] = str(kw_val)
                        serializable_config[k] = serializable_kwargs
                    else:
                        if callable(v):
                            serializable_config[k] = v.__name__ if hasattr(v, '__name__') else str(v)
                        else:
                            try:
                                json.dumps(v)  # 尝试序列化
                                serializable_config[k] = v
                            except:
                                serializable_config[k] = str(v)
                
                json.dump(serializable_config, f, indent=2)
        except Exception as e:
            print(f"警告：无法保存训练配置：{e}")
        
        # 开始训练
        train_advanced_rl(config) 