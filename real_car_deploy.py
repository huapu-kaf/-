"""
激光车辆强化学习 - 实车部署脚本
将训练好的模型部署到实车上，提供旋转角度和线速度等控制参数
"""

import os
import time
import argparse
import numpy as np
import json
from stable_baselines3 import PPO
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any

# 实车通信模块（根据实际硬件接口调整）
class RealCarInterface:
    """
    实车通信接口类，负责将模型输出转换为实车控制命令
    并从实车获取传感器数据作为模型输入
    """
    def __init__(self, 
                 serial_port: str = '/dev/ttyUSB0', 
                 baud_rate: int = 115200, 
                 connect_retry: int = 3):
        """
        初始化实车通信接口
        
        参数:
            serial_port: 串口设备路径
            baud_rate: 波特率
            connect_retry: 连接重试次数
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.connect_retry = connect_retry
        self.connected = False
        
        # 状态映射参数
        self.distance_sensors = [0.0] * 16  # 16个距离传感器读数
        self.imu_data = [0.0] * 6  # 3轴加速度和3轴角速度
        self.current_velocity = [0.0, 0.0, 0.0]  # 当前速度 (x, y, z)
        self.current_orientation = [0.0, 0.0, 0.0]  # 当前方向 (roll, pitch, yaw)
        
        # 控制参数
        self.max_linear_velocity = 1.0  # 最大线速度 m/s
        self.max_angular_velocity = 1.5  # 最大角速度 rad/s
        
        # 尝试连接设备
        self._connect()
    
    def _connect(self):
        """尝试连接实车设备"""
        try:
            # 这里根据实际硬件接口实现通信代码
            # 例如使用pyserial库连接串口
            print(f"连接实车: {self.serial_port}，波特率: {self.baud_rate}")
            # 模拟连接成功
            self.connected = True
            print("实车连接成功")
        except Exception as e:
            print(f"实车连接失败: {e}")
    
    def get_observation(self) -> np.ndarray:
        """
        从实车获取观测数据
        
        返回:
            observation: 转换为模型输入格式的观测数据
        """
        if not self.connected:
            print("警告: 实车未连接，返回模拟数据")
            return np.zeros(26)  # 返回全零数据
        
        try:
            # 这里根据实际硬件接口实现数据获取代码
            # 模拟传感器读数
            self._update_sensor_data()
            
            # 构建观测数据 (与训练环境格式相同)
            observation = np.concatenate([
                self.distance_sensors,  # 16个激光传感器距离
                self.current_velocity,  # 当前速度
                self.current_orientation,  # 当前方向
                self.imu_data  # IMU数据
            ])
            
            return observation
        except Exception as e:
            print(f"获取传感器数据失败: {e}")
            return np.zeros(26)
    
    def _update_sensor_data(self):
        """模拟更新传感器数据"""
        # 实际应用中，这里会从实车获取真实数据
        # 这里仅做测试示例，生成随机数据
        self.distance_sensors = [np.random.uniform(0.1, 5.0) for _ in range(16)]
        self.imu_data = [np.random.uniform(-1.0, 1.0) for _ in range(6)]
        self.current_velocity = [np.random.uniform(-0.5, 0.5) for _ in range(3)]
        self.current_orientation = [np.random.uniform(-0.2, 0.2) for _ in range(3)]
    
    def apply_action(self, action: np.ndarray):
        """
        将模型输出的动作应用到实车上
        
        参数:
            action: 模型输出的动作，通常是2个浮点数 [左轮速度, 右轮速度]
        """
        if not self.connected:
            print("警告: 实车未连接，无法应用动作")
            return
        
        try:
            # 将动作[-1, 1]范围映射到实际速度
            left_speed = action[0] * self.max_linear_velocity
            right_speed = action[1] * self.max_linear_velocity
            
            # 计算线速度和角速度
            linear_velocity = (left_speed + right_speed) / 2
            angular_velocity = (right_speed - left_speed) / 0.15  # 假设轮距为0.15m
            
            # 限制速度范围
            linear_velocity = np.clip(linear_velocity, -self.max_linear_velocity, self.max_linear_velocity)
            angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
            
            # 发送控制命令到实车
            print(f"发送控制: 线速度={linear_velocity:.2f}m/s, 角速度={angular_velocity:.2f}rad/s")
            # 这里根据实际硬件接口实现控制命令发送代码
        except Exception as e:
            print(f"发送控制命令失败: {e}")
    
    def close(self):
        """关闭连接"""
        if self.connected:
            # 停止所有电机
            self.apply_action([0, 0])
            # 关闭连接
            print("关闭实车连接")
            self.connected = False

# 简化的环境类，用于实车部署
class RealCarEnv:
    """
    实车环境封装类，提供与训练环境相同的接口
    """
    def __init__(self, 
                 car_interface: RealCarInterface,
                 observation_history_len: int = 4):
        """
        初始化实车环境
        
        参数:
            car_interface: 实车通信接口
            observation_history_len: 观测历史长度
        """
        self.car = car_interface
        self.observation_history_len = observation_history_len
        self.observation_history = []
        
        # 定义动作空间和观测空间
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # 观测空间与训练环境一致
        self.observation_space = gym.spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(26,), dtype=np.float32
        )
    
    def reset(self):
        """
        重置环境
        
        返回:
            observation: 初始观测
        """
        # 清空观测历史
        self.observation_history = []
        
        # 获取当前观测
        observation = self.car.get_observation()
        
        # 初始化观测历史
        for _ in range(self.observation_history_len):
            self.observation_history.append(observation)
        
        return observation
    
    def step(self, action):
        """
        执行动作并返回下一个状态
        
        参数:
            action: 模型输出的动作 [左轮速度, 右轮速度]
            
        返回:
            observation: 新的观测
            reward: 奖励 (实车部署中通常不使用)
            done: 是否结束
            info: 额外信息
        """
        # 应用动作到实车
        self.car.apply_action(action)
        
        # 等待一小段时间
        time.sleep(0.05)  # 控制频率约20Hz
        
        # 获取新的观测
        observation = self.car.get_observation()
        
        # 更新观测历史
        self.observation_history.append(observation)
        if len(self.observation_history) > self.observation_history_len:
            self.observation_history.pop(0)
        
        # 实车部署中没有奖励和结束信号
        reward = 0.0
        done = False
        info = {}
        
        return observation, reward, done, info
    
    def close(self):
        """关闭环境"""
        self.car.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="激光车辆强化学习实车部署工具")
    parser.add_argument('--model_path', type=str, default='./advanced_logs/final_model',
                      help='训练好的模型路径')
    parser.add_argument('--serial_port', type=str, default='/dev/ttyUSB0',
                      help='实车串口设备 (Windows下如COM3)')
    parser.add_argument('--baud_rate', type=int, default=115200,
                      help='串口波特率')
    parser.add_argument('--max_linear_velocity', type=float, default=1.0,
                      help='最大线速度 (m/s)')
    parser.add_argument('--max_angular_velocity', type=float, default=1.5,
                      help='最大角速度 (rad/s)')
    parser.add_argument('--control_frequency', type=float, default=20.0,
                      help='控制频率 (Hz)')
    parser.add_argument('--sim_mode', action='store_true',
                      help='使用模拟模式 (无需实际硬件)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path + ".zip"):
        print(f"错误: 模型文件 {args.model_path}.zip 不存在")
        return
    
    print(f"加载模型: {args.model_path}")
    try:
        # 加载训练好的模型
        model = PPO.load(args.model_path)
        print("模型加载成功")
        
        # 初始化实车接口
        if args.sim_mode:
            print("使用模拟模式")
            car = RealCarInterface(serial_port="SIM", baud_rate=args.baud_rate)
        else:
            car = RealCarInterface(serial_port=args.serial_port, baud_rate=args.baud_rate)
        
        # 最大速度设置
        car.max_linear_velocity = args.max_linear_velocity
        car.max_angular_velocity = args.max_angular_velocity
        
        # 创建实车环境
        env = RealCarEnv(car)
        
        # 控制循环时间间隔
        control_interval = 1.0 / args.control_frequency
        
        # 初始化环境
        observation = env.reset()
        
        print("开始实车控制循环，按Ctrl+C退出")
        try:
            while True:
                start_time = time.time()
                
                # 获取模型预测的动作
                action, _states = model.predict(observation, deterministic=True)
                
                # 应用动作到实车
                observation, reward, done, info = env.step(action)
                
                # 计算循环耗时
                elapsed = time.time() - start_time
                
                # 保持控制频率稳定
                if elapsed < control_interval:
                    time.sleep(control_interval - elapsed)
                
                # 显示信息
                print(f"动作: [{action[0]:.2f}, {action[1]:.2f}], "
                      f"频率: {1.0/(time.time() - start_time):.1f}Hz")
                
        except KeyboardInterrupt:
            print("用户中断，停止控制")
        
        # 关闭环境和实车连接
        env.close()
        print("实车控制系统已关闭")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 