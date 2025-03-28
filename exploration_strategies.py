"""
激光车辆强化学习 - 混合探索策略
包括随机网络蒸馏(RND)、对手影子训练和优先经验回放
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple
import copy
import ray

class RandomNetworkDistillation:
    """随机网络蒸馏(RND)模块 - 用于好奇心驱动探索"""
    
    def __init__(self, input_size, hidden_size=128, output_size=64, learning_rate=0.001):
        """
        初始化RND模块
        
        参数:
            input_size: 输入状态维度
            hidden_size: 隐藏层大小
            output_size: 输出特征维度
            learning_rate: 学习率
        """
        # 目标网络 - 固定随机初始化的网络
        self.target_network = self._build_network(input_size, hidden_size, output_size)
        self.target_network.eval()  # 设置为评估模式，不会更新
        
        # 预测网络 - 训练来预测目标网络的输出
        self.predictor_network = self._build_network(input_size, hidden_size, output_size)
        
        # 优化器
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=learning_rate)
        
        # 归一化统计
        self.rnd_running_mean = 0
        self.rnd_running_std = 1
        self.num_samples = 0
        self.normalize_factor = 1.0
    
    def _build_network(self, input_size, hidden_size, output_size):
        """构建RND网络"""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def get_intrinsic_reward(self, state):
        """
        计算内在奖励
        
        参数:
            state: 状态向量 (numpy数组)
            
        返回:
            float: 内在奖励值
        """
        # 转换为tensor
        state = torch.FloatTensor(state)
        
        with torch.no_grad():
            # 目标网络输出
            target_output = self.target_network(state)
        
        # 预测网络输出
        predictor_output = self.predictor_network(state)
        
        # 计算预测误差 (MSE)
        error = F.mse_loss(predictor_output, target_output).item()
        
        # 更新归一化统计
        self.num_samples += 1
        delta = error - self.rnd_running_mean
        self.rnd_running_mean += delta / self.num_samples
        delta2 = error - self.rnd_running_mean
        self.rnd_running_std += delta * delta2
        
        if self.num_samples > 1:
            self.normalize_factor = self.rnd_running_std / (self.num_samples - 1)
            if self.normalize_factor < 1e-8:
                self.normalize_factor = 1.0
        
        # 归一化后的奖励
        intrinsic_reward = error / (self.normalize_factor ** 0.5 + 1e-8)
        
        return intrinsic_reward
    
    def update(self, state):
        """
        更新预测网络
        
        参数:
            state: 状态向量 (numpy数组)
        """
        # 转换为tensor
        state = torch.FloatTensor(state)
        
        # 训练预测网络
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            target_output = self.target_network(state)
        
        predictor_output = self.predictor_network(state)
        loss = F.mse_loss(predictor_output, target_output)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class ShadowOpponent:
    """影子对手类 - 用于自我对抗训练"""
    
    def __init__(self, clone_interval=10000):
        """
        初始化影子对手
        
        参数:
            clone_interval: 克隆策略的间隔步数
        """
        self.shadow_policy = None  # 影子策略
        self.clone_interval = clone_interval
        self.step_counter = 0
    
    def update(self, current_policy):
        """
        更新影子对手策略
        
        参数:
            current_policy: 当前策略模型
        """
        self.step_counter += 1
        
        # 每clone_interval步克隆一次当前策略
        if self.step_counter % self.clone_interval == 0:
            self.shadow_policy = copy.deepcopy(current_policy)
            print(f"已更新影子对手策略，步数: {self.step_counter}")
    
    def get_action(self, state):
        """
        获取影子对手的动作
        
        参数:
            state: 当前状态
            
        返回:
            numpy数组: 影子对手的动作
        """
        if self.shadow_policy is None:
            # 如果影子策略不存在，则返回随机动作
            return np.random.uniform(-1, 1, 2)  # 假设动作是二维的[左轮速度, 右轮速度]
        
        return self.shadow_policy.predict(state)


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先经验回放缓冲区
        
        参数:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0表示均匀采样，1表示完全按优先级)
            beta: 重要性采样指数 (0表示不补偿，1表示完全补偿)
            beta_increment: beta增长率
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # 定义经验元组
        self.Experience = namedtuple('Experience', 
                                    ['state', 'action', 'reward', 'next_state', 'done'])
    
    def add(self, state, action, reward, next_state, done):
        """
        添加新经验
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 创建经验元组
        experience = self.Experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            
        # 新经验的优先级设为最大值
        self.priorities[self.position] = self.max_priority
        
        # 更新位置
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        采样一批经验
        
        参数:
            batch_size: 批量大小
            
        返回:
            tuple: (经验批次, 重要性权重, 采样索引)
        """
        if len(self.memory) < batch_size:
            return None, None, None
        
        # 更新beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算采样概率
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # 采样索引
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # 归一化
        
        # 获取经验
        batch = [self.memory[idx] for idx in indices]
        
        return batch, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """
        更新优先级
        
        参数:
            indices: 要更新的经验索引
            td_errors: TD误差(绝对值)
        """
        for idx, error in zip(indices, td_errors):
            # 避免优先级为0
            priority = max(abs(error) + 1e-5, 1e-5)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """返回缓冲区中经验的数量"""
        return len(self.memory)


class MixedExplorationStrategy:
    """混合探索策略类 - 整合RND、影子对手和优先经验回放"""
    
    def __init__(self, state_size, action_size, config=None):
        """
        初始化混合探索策略
        
        参数:
            state_size: 状态空间维度
            action_size: 动作空间维度
            config: 配置参数
        """
        # 默认配置
        self.config = {
            'rnd_enabled': True,
            'shadow_opponent_enabled': True,
            'per_enabled': True,
            'curiosity_weight': 0.01,
            'shadow_prob': 0.5,  # 使用影子对手的概率
            'buffer_capacity': 100000,
            'batch_size': 64,
            'clone_interval': 10000
        }
        
        # 更新自定义配置
        if config:
            self.config.update(config)
        
        # 初始化RND模块
        if self.config['rnd_enabled']:
            self.rnd = RandomNetworkDistillation(input_size=state_size)
        else:
            self.rnd = None
        
        # 初始化影子对手
        if self.config['shadow_opponent_enabled']:
            self.shadow_opponent = ShadowOpponent(
                clone_interval=self.config['clone_interval']
            )
        else:
            self.shadow_opponent = None
        
        # 初始化优先经验回放缓冲区
        if self.config['per_enabled']:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config['buffer_capacity']
            )
        else:
            self.replay_buffer = None
    
    def get_intrinsic_reward(self, state):
        """
        获取内在奖励
        
        参数:
            state: 状态向量
            
        返回:
            float: 内在奖励
        """
        if self.rnd:
            return self.config['curiosity_weight'] * self.rnd.get_intrinsic_reward(state)
        return 0.0
    
    def should_use_shadow_opponent(self):
        """
        决定是否使用影子对手
        
        返回:
            bool: 是否使用影子对手
        """
        if self.shadow_opponent and self.shadow_opponent.shadow_policy:
            return random.random() < self.config['shadow_prob']
        return False
    
    def update_shadow_opponent(self, current_policy):
        """
        更新影子对手
        
        参数:
            current_policy: 当前策略
        """
        if self.shadow_opponent:
            self.shadow_opponent.update(current_policy)
    
    def add_experience(self, state, action, reward, next_state, done):
        """
        添加经验到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        if self.replay_buffer:
            self.replay_buffer.add(state, action, reward, next_state, done)
    
    def sample_batch(self, batch_size=None):
        """
        从回放缓冲区采样一批经验
        
        参数:
            batch_size: 批量大小，如果为None则使用默认值
            
        返回:
            tuple: (经验批次, 重要性权重, 采样索引)
        """
        if self.replay_buffer:
            if batch_size is None:
                batch_size = self.config['batch_size']
            return self.replay_buffer.sample(batch_size)
        return None, None, None
    
    def update_priorities(self, indices, td_errors):
        """
        更新经验优先级
        
        参数:
            indices: 经验索引
            td_errors: TD误差
        """
        if self.replay_buffer:
            self.replay_buffer.update_priorities(indices, td_errors)
    
    def update_rnd(self, state):
        """
        更新RND模块
        
        参数:
            state: 状态向量
        """
        if self.rnd:
            return self.rnd.update(state)
        return 0.0 