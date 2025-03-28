"""
激光车辆强化学习 - 课程学习管理器
实现六阶段渐进式训练课程，从简单到复杂
"""

import numpy as np
import math
import random
from collections import deque
import pybullet as p

class CurriculumManager:
    """课程学习管理器 - 管理训练的不同阶段"""
    
    def __init__(self, arena, config=None):
        """
        初始化课程管理器
        
        参数:
            arena: 战斗场地对象
            config: 配置参数
        """
        self.arena = arena
        
        # 默认配置
        self.config = {
            'promotion_threshold': 0.75,  # 晋升阈值 (胜率)
            'demotion_threshold': 0.40,   # 降级阈值 (胜率)
            'promotion_window': 10,       # 晋升窗口 (局数)
            'demotion_window': 5,         # 降级窗口 (局数)
            'initial_stage': 1,           # 初始阶段
            'max_stage': 6,               # 最大阶段
            'stage_names': [              # 阶段名称
                "静止目标（无雷区）",
                "移动目标（匀速）",
                "简单规则对手",
                "添加雷区",
                "对抗性对手",
                "全随机干扰"
            ]
        }
        
        # 更新自定义配置
        if config:
            self.config.update(config)
        
        # 当前阶段
        self.current_stage = self.config['initial_stage']
        
        # 胜率记录
        self.win_history = deque(maxlen=max(self.config['promotion_window'], 
                                           self.config['demotion_window']))
        
        # 阶段特定参数
        self.stage_params = self._init_stage_params()
        
        # 调试信息
        self.debug_text_id = None
    
    def _init_stage_params(self):
        """初始化各阶段参数"""
        return {
            # 阶段1：静止目标（无雷区）
            1: {
                'opponent_type': 'static',          # 静止目标
                'mines_enabled': False,             # 无雷区
                'opponent_difficulty': 0.0,         # 最低难度
                'noise_level': 0.0,                 # 无噪声
                'victory_threshold': 2.0,           # 激光照射2秒算胜利
                'episode_timeout': 60.0,            # 60秒超时
                'reward_scale': 1.0                 # 标准奖励
            },
            # 阶段2：移动目标（匀速）
            2: {
                'opponent_type': 'uniform_motion',  # 匀速运动
                'mines_enabled': False,             # 无雷区
                'opponent_difficulty': 0.2,         # 低难度
                'noise_level': 0.0,                 # 无噪声
                'victory_threshold': 2.0,           # 激光照射2秒算胜利
                'episode_timeout': 60.0,            # 60秒超时
                'reward_scale': 1.0                 # 标准奖励
            },
            # 阶段3：简单规则对手
            3: {
                'opponent_type': 'simple_rules',    # 简单规则对手
                'mines_enabled': False,             # 无雷区
                'opponent_difficulty': 0.4,         # 中低难度
                'noise_level': 0.1,                 # 低噪声
                'victory_threshold': 2.5,           # 激光照射2.5秒算胜利
                'episode_timeout': 90.0,            # 90秒超时
                'reward_scale': 1.2                 # 稍微增强奖励
            },
            # 阶段4：添加雷区
            4: {
                'opponent_type': 'simple_rules',    # 简单规则对手
                'mines_enabled': True,              # 有雷区
                'opponent_difficulty': 0.5,         # 中等难度
                'noise_level': 0.2,                 # 中低噪声
                'victory_threshold': 2.5,           # 激光照射2.5秒算胜利
                'episode_timeout': 90.0,            # 90秒超时
                'reward_scale': 1.5                 # 增强奖励
            },
            # 阶段5：对抗性对手
            5: {
                'opponent_type': 'adversarial',     # 对抗性对手
                'mines_enabled': True,              # 有雷区
                'opponent_difficulty': 0.7,         # 高难度
                'noise_level': 0.3,                 # 中等噪声
                'victory_threshold': 3.0,           # 激光照射3秒算胜利
                'episode_timeout': 120.0,           # 120秒超时
                'reward_scale': 1.8                 # 显著增强奖励
            },
            # 阶段6：全随机干扰
            6: {
                'opponent_type': 'adversarial',     # 对抗性对手
                'mines_enabled': True,              # 有雷区
                'opponent_difficulty': 1.0,         # 最高难度
                'noise_level': 0.5,                 # 高噪声
                'random_start': True,               # 随机起始位置
                'random_field': True,               # 随机场地布局
                'victory_threshold': 3.0,           # 激光照射3秒算胜利
                'episode_timeout': 120.0,           # 120秒超时
                'reward_scale': 2.0                 # 最大奖励
            }
        }
    
    def setup_environment(self):
        """根据当前阶段设置环境"""
        stage = self.current_stage
        params = self.stage_params[stage]
        
        # 清理旧的雷区
        if hasattr(self.arena, 'mines'):
            for mine in self.arena.mines:
                p.removeBody(mine)
            self.arena.mines = []
        
        # 设置雷区
        if params['mines_enabled']:
            if 'random_field' in params and params['random_field']:
                # 随机雷区位置
                self.arena.mine_positions = self._generate_random_mine_positions()
            
            # 创建雷区
            self.arena._create_mines()
        
        # 显示当前课程阶段
        self._update_stage_display()
        
        return params
    
    def _generate_random_mine_positions(self):
        """生成随机雷区位置"""
        # 场地边界范围内随机生成3个不重叠的雷区
        mines = []
        attempts = 0
        max_attempts = 100
        
        while len(mines) < 3 and attempts < max_attempts:
            # 随机位置 (在场地中心区域)
            x = random.uniform(-1.5, 1.5)
            y = random.uniform(-1.5, 1.5)
            
            # 检查是否与已有雷区重叠
            overlap = False
            for mx, my, _ in mines:
                dist = math.sqrt((x - mx) ** 2 + (y - my) ** 2)
                if dist < 2 * self.arena.mine_radius:  # 避免重叠
                    overlap = True
                    break
            
            if not overlap:
                mines.append((x, y, self.arena.mine_height/2))
            
            attempts += 1
        
        # 如果未能生成3个，使用默认布局补齐
        default_positions = [
            (-1.0, 0, self.arena.mine_height/2),
            (0, 0, self.arena.mine_height/2),
            (1.0, 0, self.arena.mine_height/2)
        ]
        
        while len(mines) < 3:
            mines.append(default_positions[len(mines)])
        
        return mines
    
    def _update_stage_display(self):
        """更新阶段显示"""
        stage_name = self.config['stage_names'][self.current_stage - 1]
        text = f"当前训练阶段: {self.current_stage} - {stage_name}"
        
        if self.debug_text_id is None:
            self.debug_text_id = p.addUserDebugText(
                text,
                textPosition=[0, -1.8, 0.5],
                textColorRGB=[1, 1, 0],
                textSize=1.2
            )
        else:
            p.addUserDebugText(
                text,
                textPosition=[0, -1.8, 0.5],
                textColorRGB=[1, 1, 0],
                textSize=1.2,
                replaceItemUniqueId=self.debug_text_id
            )
    
    def get_opponent_action(self, opponent_vehicle, agent_vehicle):
        """
        根据当前阶段获取对手动作
        
        参数:
            opponent_vehicle: 对手车辆
            agent_vehicle: 智能体车辆
            
        返回:
            numpy数组: 对手动作 [左轮速度, 右轮速度]
        """
        stage = self.current_stage
        params = self.stage_params[stage]
        
        # 获取状态信息
        agent_state = agent_vehicle.get_state()
        opponent_state = opponent_vehicle.get_state()
        
        # 基于对手类型生成动作
        if params['opponent_type'] == 'static':
            # 静止目标
            return np.array([0.0, 0.0])
        
        elif params['opponent_type'] == 'uniform_motion':
            # 匀速运动 (沿简单轨迹)
            t = p.getTimeStep() * p.getPhysicsEngineParameters()['numSubSteps']
            period = 10.0  # 周期
            phase = (t % period) / period
            
            if phase < 0.25:
                # 前进
                return np.array([0.5, 0.5])
            elif phase < 0.5:
                # 右转
                return np.array([0.5, 0.0])
            elif phase < 0.75:
                # 前进
                return np.array([0.5, 0.5])
            else:
                # 左转
                return np.array([0.0, 0.5])
        
        elif params['opponent_type'] == 'simple_rules':
            # 简单规则对手
            # 1. 躲避智能体的激光
            # 2. 尝试接近智能体但保持安全距离
            # 3. 避开雷区
            
            # 获取相对位置和距离
            rel_pos = agent_state['self']['position'] - opponent_state['self']['position']
            distance = np.linalg.norm(rel_pos[:2])  # 水平距离
            
            # 获取对手朝向
            opponent_orient = opponent_state['self']['orientation']
            
            # 计算到智能体的角度
            angle_to_agent = math.atan2(rel_pos[1], rel_pos[0])
            
            # 计算角度差
            angle_diff = angle_to_agent - opponent_orient
            # 归一化到 [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # 基础动作
            action = np.zeros(2)
            
            # 根据角度差调整朝向
            if abs(angle_diff) > 0.2:
                if angle_diff > 0:
                    # 左转
                    action = np.array([0.2, 0.8])
                else:
                    # 右转
                    action = np.array([0.8, 0.2])
            else:
                # 前进或后退，取决于距离
                if distance > 1.5:
                    # 前进接近
                    action = np.array([0.7, 0.7])
                elif distance < 1.0:
                    # 后退远离
                    action = np.array([-0.5, -0.5])
                else:
                    # 保持距离，围绕移动
                    action = np.array([0.6, 0.4])
            
            # 雷区避障
            if params['mines_enabled']:
                mine_distance = opponent_state['environment']['mine_distance']
                if mine_distance < 0.5:
                    # 远离雷区的紧急规避
                    # 使用光线投射找出安全方向
                    safe_direction = self._find_safe_direction(opponent_vehicle)
                    action = safe_direction
            
            # 随机噪声
            noise = np.random.normal(0, params['noise_level'], 2)
            action = np.clip(action + noise, -1.0, 1.0)
            
            return action
        
        elif params['opponent_type'] == 'adversarial':
            # 对抗性对手 - 主动追踪并攻击智能体
            
            # 获取相对位置和距离
            rel_pos = agent_state['self']['position'] - opponent_state['self']['position']
            distance = np.linalg.norm(rel_pos[:2])  # 水平距离
            
            # 获取当前朝向
            opponent_orient = opponent_state['self']['orientation']
            
            # 计算到智能体的角度
            angle_to_agent = math.atan2(rel_pos[1], rel_pos[0])
            
            # 计算角度差
            angle_diff = angle_to_agent - opponent_orient
            # 归一化到 [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # 基础动作
            action = np.zeros(2)
            
            # 判断智能体是否在激光范围内
            in_laser_range = False
            if abs(angle_diff) < math.radians(30) and distance < 3.0:
                in_laser_range = True
            
            if in_laser_range:
                # 如果已经瞄准，保持方向，调整距离
                if distance > 2.0:
                    # 接近目标
                    action = np.array([0.8, 0.8])
                elif distance < 1.0:
                    # 远离目标
                    action = np.array([-0.5, -0.5])
                else:
                    # 保持距离
                    action = np.array([0.1, 0.1])
            else:
                # 调整朝向以瞄准智能体
                if angle_diff > 0:
                    # 左转
                    turn_rate = min(0.5, abs(angle_diff) / math.pi)
                    action = np.array([0.5 - turn_rate, 0.5 + turn_rate])
                else:
                    # 右转
                    turn_rate = min(0.5, abs(angle_diff) / math.pi)
                    action = np.array([0.5 + turn_rate, 0.5 - turn_rate])
            
            # 雷区避障
            if params['mines_enabled']:
                mine_distance = opponent_state['environment']['mine_distance']
                if mine_distance < 0.5:
                    # 远离雷区的紧急规避
                    safe_direction = self._find_safe_direction(opponent_vehicle)
                    
                    # 融合安全方向与当前动作
                    blend_factor = 1.0 - min(1.0, mine_distance / 0.5)  # 距离越近，安全性越重要
                    action = blend_factor * safe_direction + (1.0 - blend_factor) * action
            
            # 增加难度相关的随机性
            difficulty = params['opponent_difficulty']
            
            # 随机扰动 (难度越高，扰动越小)
            noise_scale = params['noise_level'] * (1.0 - difficulty)
            noise = np.random.normal(0, noise_scale, 2)
            
            # 偶尔的随机行为 (难度越低，越频繁)
            if random.random() < 0.05 * (1.0 - difficulty):
                action = np.random.uniform(-1, 1, 2)
            
            # 确保在有效范围内
            action = np.clip(action + noise, -1.0, 1.0)
            
            return action
        
        # 默认返回随机动作
        return np.random.uniform(-1, 1, 2)
    
    def _find_safe_direction(self, vehicle):
        """
        使用光线投射找出远离雷区的安全方向
        
        参数:
            vehicle: 车辆对象
            
        返回:
            numpy数组: 安全方向的动作
        """
        # 获取车辆位置和朝向
        state = vehicle.get_state()
        pos = state['self']['position']
        orient = state['self']['orientation']
        
        # 在不同角度投射光线检测安全方向
        angles = np.linspace(0, 2*math.pi, 8, endpoint=False)  # 8个方向
        safe_distances = []
        
        for angle in angles:
            # 计算投射方向
            ray_dir = np.array([
                math.cos(orient + angle),
                math.sin(orient + angle),
                0
            ])
            
            # 起点略微抬高以避免检测到地面
            ray_start = np.array([pos[0], pos[1], 0.05])
            ray_end = ray_start + ray_dir * 2.0  # 检测2米范围
            
            # 执行光线投射
            result = p.rayTest(ray_start, ray_end)[0]
            hit_fraction = result[2]
            
            # 如果没有击中任何物体，距离为最大值
            if hit_fraction >= 1.0:
                safe_distances.append(2.0)
            else:
                # 计算实际距离
                safe_distances.append(hit_fraction * 2.0)
        
        # 找出最安全的方向（距离最远）
        safest_idx = np.argmax(safe_distances)
        safest_angle = angles[safest_idx]
        
        # 转换为车辆动作
        # 计算与当前朝向的角度差
        angle_diff = safest_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        
        # 基于角度差生成动作
        if abs(angle_diff) < 0.2:
            # 基本上是直行
            return np.array([0.7, 0.7])
        elif angle_diff > 0:
            # 左转 (角度差越大，转弯越急)
            turn_rate = min(0.7, angle_diff / math.pi)
            return np.array([0.7 - turn_rate, 0.7 + turn_rate])
        else:
            # 右转
            turn_rate = min(0.7, -angle_diff / math.pi)
            return np.array([0.7 + turn_rate, 0.7 - turn_rate])
    
    def record_outcome(self, win):
        """
        记录对局结果
        
        参数:
            win: 布尔值，True表示智能体获胜
        """
        self.win_history.append(1 if win else 0)
        
        # 检查是否需要升级或降级
        self._check_for_promotion_or_demotion()
    
    def _check_for_promotion_or_demotion(self):
        """检查是否需要升级或降级"""
        # 至少需要达到窗口大小才能评估
        if len(self.win_history) < min(self.config['promotion_window'], 
                                      self.config['demotion_window']):
            return
        
        # 计算最近N局的胜率
        if len(self.win_history) >= self.config['promotion_window']:
            recent_wins = list(self.win_history)[-self.config['promotion_window']:]
            promotion_win_rate = sum(recent_wins) / len(recent_wins)
            
            # 检查是否达到晋升条件
            if promotion_win_rate >= self.config['promotion_threshold']:
                self._promote()
                return
        
        # 计算最近M局的胜率
        if len(self.win_history) >= self.config['demotion_window']:
            recent_wins = list(self.win_history)[-self.config['demotion_window']:]
            demotion_win_rate = sum(recent_wins) / len(recent_wins)
            
            # 检查是否需要降级
            if demotion_win_rate <= self.config['demotion_threshold']:
                self._demote()
                return
    
    def _promote(self):
        """晋升到下一阶段"""
        if self.current_stage < self.config['max_stage']:
            self.current_stage += 1
            self.win_history.clear()  # 清空历史记录
            print(f"晋升到第 {self.current_stage} 阶段: {self.config['stage_names'][self.current_stage - 1]}")
            self.setup_environment()  # 重新设置环境
        else:
            print("已经达到最高阶段!")
    
    def _demote(self):
        """降级到前一阶段"""
        if self.current_stage > 1:
            self.current_stage -= 1
            self.win_history.clear()  # 清空历史记录
            print(f"降级到第 {self.current_stage} 阶段: {self.config['stage_names'][self.current_stage - 1]}")
            self.setup_environment()  # 重新设置环境
        else:
            print("已经在最低阶段!")
    
    def get_current_stage_params(self):
        """获取当前阶段的参数配置"""
        return self.stage_params[self.current_stage]
    
    def reset(self):
        """重置课程管理器到初始状态"""
        self.current_stage = self.config['initial_stage']
        self.win_history.clear()
        self.setup_environment()  # 重新设置环境 