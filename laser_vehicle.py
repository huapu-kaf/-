import pybullet as p
import numpy as np
import math
import time
from collections import deque

class LaserVehicle:
    """
    带激光检测的差速驱动车辆模型
    """
    
    def __init__(self, client_id, start_pos, start_orientation=(0, 0, 0), 
                 vehicle_size=(0.2, 0.15, 0.1), mass=1.5, color=[0.1, 0.1, 0.8, 1]):
        """
        初始化激光车辆
        
        参数:
            client_id: PyBullet 物理引擎客户端ID
            start_pos: 起始位置 (x, y, z)
            start_orientation: 起始朝向 (roll, pitch, yaw)，单位：弧度
            vehicle_size: 车辆尺寸 (length, width, height)，单位：米
            mass: 车辆质量，单位：kg
            color: 车辆颜色 [r, g, b, a]
        """
        self.client_id = client_id
        self.size = vehicle_size
        self.mass = mass
        self.color = color
        
        # 物理约束参数 - 提高速度和灵活性
        self.max_velocity = 2.0  # 最大线速度 (m/s)，从1.2提高到2.0
        self.max_angular_velocity = math.pi * 1.5  # 最大角速度 (rad/s)，从π提高到1.5π
        self.max_acceleration = 2.5  # 最大加速度 (m/s²)，从1.5提高到2.5
        
        # 激光参数
        self.laser_angle_h = math.radians(60)  # 水平激光角度范围（±60度）
        self.laser_angle_v = math.radians(15)  # 垂直激光角度范围（±15度）
        self.laser_detect_frequency = 30  # 激光检测频率（Hz）
        self.laser_max_distance = 5.0  # 激光最大检测距离
        
        # 激光系统参数 - 修改为无冷却时间，可以持续发射
        self.laser_cooldown_time = 0.0  # 激光冷却时间设为0秒，可持续发射
        self.last_fire_time = 0  # 上次开火时间
        self.laser_active = False  # 激光是否激活
        self.laser_particles = []  # 激光粒子效果
        self.laser_damage = 1.0  # 激光伤害
        self.laser_lines = []  # 激光可视化线条
        
        # 创建车辆
        self.vehicle_id = self._create_vehicle(start_pos, start_orientation)
        
        # 记录最近一段时间内被激光照射的累计时间
        self.laser_hit_times = deque(maxlen=int(0.5 * self.laser_detect_frequency))
        self.last_laser_check_time = time.time()
        
        # 对手车辆ID，默认为None
        self.opponent_id = None
        
        # 雷区信息
        self.mine_positions = []
        self.mine_radius = 0.5
    
    def _create_vehicle(self, position, orientation):
        """创建车辆物理模型，包括白色漫反射区域"""
        # 获取车辆尺寸
        length, width, height = self.size
        
        # 创建主体碰撞形状
        collision_shape_id = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[length/2, width/2, height/2]
        )
        
        # 创建主体视觉形状
        visual_shape_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[length/2, width/2, height/2],
            rgbaColor=self.color  # 使用传入的颜色
        )
        
        # 计算转动惯量（长方体公式）
        inertia_x = (1/12) * self.mass * (width**2 + height**2)
        inertia_y = (1/12) * self.mass * (length**2 + height**2)
        inertia_z = (1/12) * self.mass * (length**2 + width**2)
        
        # 创建车辆多体
        vehicle_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler(orientation),
            baseInertialFramePosition=[0, 0, 0],
            baseInertialFrameOrientation=[0, 0, 0, 1]
        )
        
        # 添加白色漫反射区域（侧面涂装，高5cm）
        reflection_height = 0.05  # 漫反射区域高度5cm
        reflection_z_pos = height/2 - reflection_height/2  # 位于车辆侧面的中间位置
        
        # 左侧白色漫反射区域
        left_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length/2, 0.001, reflection_height/2],  # 很薄的盒子贴在左侧
            rgbaColor=[1.0, 1.0, 1.0, 1.0]  # 纯白色
        )
        
        p.createMultiBody(
            baseMass=0,  # 无质量，只是视觉效果
            baseCollisionShapeIndex=-1,  # 无碰撞形状
            baseVisualShapeIndex=left_visual,
            basePosition=[position[0], position[1] - width/2, position[2]],
            baseOrientation=p.getQuaternionFromEuler(orientation)
        )
        
        # 右侧白色漫反射区域
        right_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length/2, 0.001, reflection_height/2],  # 很薄的盒子贴在右侧
            rgbaColor=[1.0, 1.0, 1.0, 1.0]  # 纯白色
        )
        
        p.createMultiBody(
            baseMass=0,  # 无质量，只是视觉效果
            baseCollisionShapeIndex=-1,  # 无碰撞形状
            baseVisualShapeIndex=right_visual,
            basePosition=[position[0], position[1] + width/2, position[2]],
            baseOrientation=p.getQuaternionFromEuler(orientation)
        )
        
        # 前侧白色漫反射区域
        front_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.001, width/2, reflection_height/2],  # 很薄的盒子贴在前侧
            rgbaColor=[1.0, 1.0, 1.0, 1.0]  # 纯白色
        )
        
        p.createMultiBody(
            baseMass=0,  # 无质量，只是视觉效果
            baseCollisionShapeIndex=-1,  # 无碰撞形状
            baseVisualShapeIndex=front_visual,
            basePosition=[position[0] + length/2, position[1], position[2]],
            baseOrientation=p.getQuaternionFromEuler(orientation)
        )
        
        # 后侧白色漫反射区域
        back_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.001, width/2, reflection_height/2],  # 很薄的盒子贴在后侧
            rgbaColor=[1.0, 1.0, 1.0, 1.0]  # 纯白色
        )
        
        p.createMultiBody(
            baseMass=0,  # 无质量，只是视觉效果
            baseCollisionShapeIndex=-1,  # 无碰撞形状
            baseVisualShapeIndex=back_visual,
            basePosition=[position[0] - length/2, position[1], position[2]],
            baseOrientation=p.getQuaternionFromEuler(orientation)
        )
        
        return vehicle_id
    
    def draw_direction_indicator(self):
        """绘制方向指示器，显示车辆的朝向"""
        # 获取当前车辆位置和朝向
        pos, quat = p.getBasePositionAndOrientation(self.vehicle_id)
        euler = p.getEulerFromQuaternion(quat)
        yaw = euler[2]
        
        # 计算箭头起点和终点
        length, width, height = self.size
        arrow_length = length * 1.2  # 增大箭头长度以便于观察
        
        arrow_start = [
            pos[0], 
            pos[1], 
            pos[2] + height/2
        ]
        
        arrow_end = [
            pos[0] + arrow_length * math.cos(yaw),
            pos[1] + arrow_length * math.sin(yaw),
            pos[2] + height/2
        ]
        
        # 调整颜色亮度以便更容易看到
        indicator_color = list(self.color[:3])  # 复制RGB部分
        # 确保颜色足够亮
        for i in range(3):
            indicator_color[i] = min(1.0, indicator_color[i] * 1.5)
        
        # 使用调试线段作为指示器（每次调用都会创建新的线段）
        p.addUserDebugLine(
            arrow_start,
            arrow_end,
            indicator_color,  # 使用调亮的车辆颜色
            lineWidth=3.0,
            lifeTime=1/30  # 只显示一帧的时间，然后需要刷新
        )
    
    def apply_action(self, action):
        """
        应用差速驱动控制动作
        
        参数:
            action: [left_wheel_velocity, right_wheel_velocity] - 左右轮速度，范围[-1, 1]
        """
        # 检查输入有效性
        if not isinstance(action, (list, tuple, np.ndarray)) or len(action) != 2:
            raise ValueError(f"动作必须是长度为2的列表，接收到: {action}")
        
        # 应用约束，防止速度过大
        left_wheel_vel = np.clip(action[0], -1.0, 1.0) * self.max_velocity
        right_wheel_vel = np.clip(action[1], -1.0, 1.0) * self.max_velocity
        
        # 计算线速度和角速度
        linear_vel = (left_wheel_vel + right_wheel_vel) / 2
        angular_vel = (right_wheel_vel - left_wheel_vel) / (self.size[1] * 1.1)  # 轮距略大于宽度
        
        # 限制角速度
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        # 获取当前朝向
        _, orientation = self.get_position_and_orientation()
        yaw = orientation[2]
        
        # 根据朝向计算速度分量
        vel_x = linear_vel * math.cos(yaw)
        vel_y = linear_vel * math.sin(yaw)
        
        # 设置车辆速度
        p.resetBaseVelocity(
            self.vehicle_id,
            linearVelocity=[vel_x, vel_y, 0],
            angularVelocity=[0, 0, angular_vel],
            physicsClientId=self.client_id
        )
        
        # 持续发射激光（无冷却时间）
        self.fire_laser()
        
        # 绘制朝向指示器
        self.draw_direction_indicator()
    
    def set_opponent(self, opponent_id):
        """设置对手车辆ID"""
        self.opponent_id = opponent_id
    
    def set_mine_info(self, mine_positions, mine_radius):
        """设置雷区信息"""
        self.mine_positions = mine_positions
        self.mine_radius = mine_radius
    
    def check_laser_hit(self):
        """
        检查激光是否击中对手
        
        返回:
            bool: 是否击中对手
        """
        if self.opponent_id is None:
            return False
        
        current_time = time.time()
        # 限制检测频率
        if current_time - self.last_laser_check_time < 1.0 / self.laser_detect_frequency:
            return False
        
        self.last_laser_check_time = current_time
        
        # 获取自身位置和朝向
        own_pos, own_quat = p.getBasePositionAndOrientation(self.vehicle_id)
        own_euler = p.getEulerFromQuaternion(own_quat)
        own_yaw = own_euler[2]
        
        # 获取对手位置
        opponent_pos, _ = p.getBasePositionAndOrientation(self.opponent_id)
        
        # 计算到对手的矢量
        to_opponent = [
            opponent_pos[0] - own_pos[0],
            opponent_pos[1] - own_pos[1],
            opponent_pos[2] - own_pos[2]
        ]
        
        # 计算水平角度（相对于自身朝向）
        horizontal_angle = math.atan2(to_opponent[1], to_opponent[0]) - own_yaw
        # 归一化到 [-pi, pi]
        while horizontal_angle > math.pi:
            horizontal_angle -= 2 * math.pi
        while horizontal_angle < -math.pi:
            horizontal_angle += 2 * math.pi
        
        # 计算垂直角度
        distance_horizontal = math.sqrt(to_opponent[0]**2 + to_opponent[1]**2)
        vertical_angle = math.atan2(to_opponent[2], distance_horizontal)
        
        # 检查是否在激光范围内
        if (abs(horizontal_angle) <= self.laser_angle_h and 
            abs(vertical_angle) <= self.laser_angle_v):
            
            # 计算距离
            distance = math.sqrt(to_opponent[0]**2 + to_opponent[1]**2 + to_opponent[2]**2)
            
            if distance <= self.laser_max_distance:
                # 执行射线检测，看是否有障碍物阻挡
                ray_start = [
                    own_pos[0], 
                    own_pos[1], 
                    own_pos[2]
                ]
                ray_end = [
                    opponent_pos[0], 
                    opponent_pos[1], 
                    opponent_pos[2]
                ]
                
                ray_result = p.rayTest(ray_start, ray_end)[0]
                hit_object_id = ray_result[0]
                
                # 如果射线首先击中的是对手，则激光命中
                if hit_object_id == self.opponent_id:
                    # 绘制激光射线（使用车辆颜色）
                    p.addUserDebugLine(
                        ray_start,
                        ray_end,
                        self.color[:3],  # 使用车辆颜色的RGB部分
                        lifeTime=0.1,  # 持续0.1秒
                        lineWidth=2.0
                    )
                    
                    # 记录击中
                    self.laser_hit_times.append(1.0 / self.laser_detect_frequency)
                    return True
        
        # 未命中时清除激光击中记录
        self.laser_hit_times.append(0)
        return False
    
    def get_laser_hit_time(self):
        """
        获取最近0.5秒内被激光照射的累计时间
        
        返回:
            float: 累计照射时间（秒）
        """
        return sum(self.laser_hit_times)
    
    def get_state(self):
        """
        获取车辆的状态信息，包括自身状态、对手状态和环境状态
        
        返回:
            dict: 包含各种状态信息的字典
        """
        # 获取自身位置、朝向和速度
        pos, quat = p.getBasePositionAndOrientation(self.vehicle_id)
        euler = p.getEulerFromQuaternion(quat)
        linear_vel, angular_vel = p.getBaseVelocity(self.vehicle_id)
        
        # 计算线速度大小
        linear_speed = math.sqrt(linear_vel[0]**2 + linear_vel[1]**2)
        
        # 自身状态
        self_state = {
            'position': np.array([pos[0], pos[1]]),  # x, y位置
            'orientation': euler[2],  # 航向角
            'linear_velocity': linear_speed,  # 线速度大小
            'angular_velocity': angular_vel[2]  # 角速度
        }
        
        # 对手状态（如果有对手）
        opponent_state = {}
        if self.opponent_id is not None:
            opp_pos, opp_quat = p.getBasePositionAndOrientation(self.opponent_id)
            opp_euler = p.getEulerFromQuaternion(opp_quat)
            opp_linear_vel, _ = p.getBaseVelocity(self.opponent_id)
            
            # 相对位置
            rel_pos = [
                opp_pos[0] - pos[0],
                opp_pos[1] - pos[1]
            ]
            
            # 相对朝向（对手朝向减去自身朝向）
            rel_orientation = opp_euler[2] - euler[2]
            # 归一化到[-π, π]范围
            while rel_orientation > math.pi:
                rel_orientation -= 2 * math.pi
            while rel_orientation < -math.pi:
                rel_orientation += 2 * math.pi
            
            # 相对速度
            rel_vel = [
                opp_linear_vel[0] - linear_vel[0],
                opp_linear_vel[1] - linear_vel[1]
            ]
            
            opponent_state = {
                'relative_position': np.array(rel_pos),  # 相对位置
                'relative_velocity': np.array(rel_vel),  # 相对速度
                'relative_orientation': rel_orientation,  # 相对朝向
                'orientation': opp_euler[2],  # 对手朝向
                'laser_hit_time': self.get_laser_hit_time()  # 激光照射时间
            }
        else:
            # 如果没有对手，设置为默认值
            opponent_state = {
                'relative_position': np.array([0.0, 0.0]),
                'relative_velocity': np.array([0.0, 0.0]),
                'relative_orientation': 0.0,  # 相对朝向默认值
                'orientation': 0.0,  # 对手朝向默认值
                'laser_hit_time': 0.0
            }
        
        # 环境状态
        env_state = {}
        
        # 计算到最近雷区边缘的距离
        min_mine_dist = float('inf')
        if self.mine_positions:
            for mine_pos in self.mine_positions:
                dx = pos[0] - mine_pos[0]
                dy = pos[1] - mine_pos[1]
                dist_to_center = math.sqrt(dx**2 + dy**2)
                dist_to_edge = max(0, dist_to_center - self.mine_radius)
                min_mine_dist = min(min_mine_dist, dist_to_edge)
        
        # 如果没有雷区信息，设置为默认值
        if min_mine_dist == float('inf'):
            min_mine_dist = 0.0
        
        # 到分界线的距离
        distance_to_border = abs(pos[1])
        
        # 是否在对方半场
        in_opponent_field = 1.0 if (
            (pos[1] > 0 and self_state['position'][0] < 0) or 
            (pos[1] < 0 and self_state['position'][0] > 0)
        ) else 0.0
        
        # 到场地边界的距离
        distance_to_x_boundary = min(2.0 - abs(pos[0]), abs(pos[0]) + 2.0)
        distance_to_y_boundary = min(2.0 - abs(pos[1]), abs(pos[1]) + 2.0)
        min_boundary_distance = min(distance_to_x_boundary, distance_to_y_boundary)
        
        env_state = {
            'mine_distance': min_mine_dist,
            'border_distance': distance_to_border,
            'in_opponent_field': in_opponent_field,
            'boundary_distance': np.array([distance_to_x_boundary, distance_to_y_boundary])
        }
        
        # 组合状态
        state = {
            'self': self_state,
            'opponent': opponent_state,
            'environment': env_state
        }
        
        return state
    
    def normalize_state(self, state):
        """
        将状态归一化到[-1, 1]范围
        
        参数:
            state: 未归一化的状态
            
        返回:
            dict: 归一化后的状态
        """
        norm_state = {}
        
        # 归一化自身状态
        norm_state['self'] = {
            # 位置归一化到[-1, 1]，场地范围是[-2, 2]
            'position': state['self']['position'] / 2.0,
            # 朝向归一化，已经在[-π, π]范围内，除以π归一化到[-1, 1]
            'orientation': state['self']['orientation'] / math.pi,
            # 线速度归一化，最大速度为self.max_velocity
            'linear_velocity': state['self']['linear_velocity'] / self.max_velocity,
            # 角速度归一化，最大角速度为self.max_angular_velocity
            'angular_velocity': state['self']['angular_velocity'] / self.max_angular_velocity
        }
        
        # 归一化对手状态
        norm_state['opponent'] = {
            # 相对位置归一化，场地对角线长度约为5.66米
            'relative_position': state['opponent']['relative_position'] / 4.0,
            # 相对速度归一化，假设最大相对速度为两倍最大速度
            'relative_velocity': state['opponent']['relative_velocity'] / (2 * self.max_velocity),
            # 相对朝向归一化，已经在[-π, π]范围内，除以π归一化到[-1, 1]
            'relative_orientation': state['opponent']['relative_orientation'] / math.pi,
            # 对手朝向归一化，已经在[-π, π]范围内，除以π归一化到[-1, 1]
            'orientation': state['opponent']['orientation'] / math.pi,
            # 激光照射时间归一化，最大时间为0.5秒
            'laser_hit_time': min(1.0, state['opponent']['laser_hit_time'] / 0.5)
        }
        
        # 归一化环境状态
        norm_state['environment'] = {
            # 雷区距离归一化，最大距离约为场地对角线长度
            'mine_distance': min(1.0, state['environment']['mine_distance'] / 4.0),
            # 分界线距离归一化，最大距离为2.0
            'border_distance': state['environment']['border_distance'] / 2.0,
            # 是否在对方半场已经是[0,1]，转换为[-1,1]
            'in_opponent_field': state['environment']['in_opponent_field'] * 2.0 - 1.0,
            # 边界距离归一化，最大距离为2.0
            'boundary_distance': state['environment']['boundary_distance'] / 2.0
        }
        
        return norm_state
    
    def get_flattened_state(self, normalized=True):
        """
        获取扁平化的状态向量（用于机器学习）
        
        参数:
            normalized: 是否返回归一化的状态
            
        返回:
            numpy.ndarray: 扁平化的状态向量（22维）
        """
        state = self.get_state()
        if normalized:
            state = self.normalize_state(state)
        
        # 提取各个状态分量
        s = state['self']
        o = state['opponent']
        e = state['environment']
        
        # 构建扁平化向量（总共22维）
        flat_state = np.concatenate([
            # 自身状态 (9维)
            s['position'],                      # 2维: x, y位置
            [s['orientation']],                 # 1维: 航向角
            [s['linear_velocity']],             # 1维: 线速度
            [s['angular_velocity']],            # 1维: 角速度
            
            # 对手状态 (8维)
            o['relative_position'],             # 2维: 相对位置
            o['relative_velocity'],             # 2维: 相对速度
            [o['relative_orientation']],        # 1维: 相对朝向
            [o['orientation']],                 # 1维: 对手朝向
            [o['laser_hit_time']],              # 1维: 激光照射时间
            
            # 环境状态 (5维)
            [e['mine_distance']],               # 1维: 雷区距离
            [e['border_distance']],             # 1维: 分界线距离
            [e['in_opponent_field']],           # 1维: 是否在对方半场
            e['boundary_distance']              # 2维: 边界距离
        ])
        
        return flat_state
    
    def fire_laser(self):
        """发射激光并可视化激光路径"""
        current_time = time.time()
        
        try:
            # 清除旧的激光线条
            for line in self.laser_lines:
                try:
                    p.removeUserDebugItem(line, physicsClientId=self.client_id)
                except:
                    pass  # 忽略删除调试项时的错误
            self.laser_lines = []
            
            # 获取当前位置和朝向
            position, orientation = self.get_position_and_orientation()
            yaw = orientation[2]
            
            # 激光起点（稍微抬高，从车顶部发射）
            laser_start = [
                position[0],
                position[1],
                position[2] + self.size[2]/2  # 从车顶部发射
            ]
            
            # 激光方向（与车辆朝向一致）
            laser_direction = [
                math.cos(yaw),
                math.sin(yaw),
                0  # 水平激光
            ]
            
            # 激光终点
            laser_end = [
                laser_start[0] + laser_direction[0] * self.laser_max_distance,
                laser_start[1] + laser_direction[1] * self.laser_max_distance,
                laser_start[2] + laser_direction[2] * self.laser_max_distance
            ]
            
            # 执行激光射线检测
            result = p.rayTest(laser_start, laser_end, physicsClientId=self.client_id)[0]
            hit_obj_id = result[0]
            hit_pos = result[3]
            
            # 如果激光射中了物体，更新终点位置为击中点
            if hit_obj_id != -1:
                laser_end = hit_pos
            
            # 可视化激光路径（绿色）
            line_id = p.addUserDebugLine(
                laser_start,
                laser_end,
                lineColorRGB=[0, 1, 0],  # 绿色激光
                lineWidth=2.0,
                lifeTime=1/30,  # 一帧的时间
                physicsClientId=self.client_id
            )
            self.laser_lines.append(line_id)
            
            # 检查是否击中对手
            if hit_obj_id != -1 and self.opponent_id is not None and hit_obj_id == self.opponent_id:
                # 记录激光击中时间
                self.laser_hit_times.append(current_time)
                # 设置激光激活状态
                self.laser_active = True
            else:
                self.laser_active = False
        
        except Exception as e:
            # 如果是连接断开错误，不再尝试执行其他操作
            if "Not connected to physics server" in str(e):
                raise  # 重新抛出错误，让调用者处理
            else:
                print(f"激光发射过程中发生错误: {e}")
    
    def get_laser_cooldown_normalized(self):
        """
        获取归一化的激光冷却状态（0-1）
        0表示完全冷却，1表示刚刚开火
        """
        current_time = time.time()
        time_since_fire = current_time - self.last_fire_time
        cooldown_progress = min(1.0, time_since_fire / self.laser_cooldown_time)
        return 1.0 - cooldown_progress
    
    def set_difficulty(self, difficulty):
        """
        根据难度调整激光系统参数
        
        参数:
            difficulty: 难度值 (0.0-1.0)
        """
        # 随难度增加而减少冷却时间
        self.laser_cooldown_time = 2.0 - difficulty  # 从2秒到1秒
        # 随难度增加而增加伤害
        self.laser_damage = 1.0 + difficulty  # 从1.0到2.0
        
        # 调整其他参数
        self.max_velocity = 2.0 + difficulty  # 速度随难度增加
        self.max_angular_velocity = math.pi * (1.5 + 0.5 * difficulty)  # 转向能力随难度增加
        self.max_acceleration = 2.5 + difficulty  # 加速度随难度增加
    
    def get_laser_distances(self):
        """
        获取激光传感器检测到的距离数据
        
        返回:
            numpy.ndarray: 长度为16的数组，表示不同角度的激光检测距离
        """
        angles = np.linspace(-self.laser_angle_h, self.laser_angle_h, 16)
        distances = np.full(16, self.laser_max_distance)
        
        position, orientation = self.get_position_and_orientation()
        orientation_quaternion = p.getQuaternionFromEuler(orientation)
        
        for i, angle in enumerate(angles):
            # 计算激光角度（绕Z轴的朝向加上当前激光角度）
            ray_orientation = np.array(p.getMatrixFromQuaternion(orientation_quaternion)).reshape(3, 3) @ np.array([1, 0, 0])
            # 计算旋转矩阵
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            # 应用旋转矩阵
            ray_direction = rotation_matrix @ ray_orientation
            
            # 执行射线检测
            from_pos = position
            to_pos = [
                position[0] + ray_direction[0] * self.laser_max_distance,
                position[1] + ray_direction[1] * self.laser_max_distance,
                position[2] + ray_direction[2] * self.laser_max_distance
            ]
            
            result = p.rayTest(from_pos, to_pos)
            if result[0][0] != -1:  # 如果射线击中了物体
                hit_fraction = result[0][2]
                distances[i] = hit_fraction * self.laser_max_distance
        
        return distances.astype(np.float32)
    
    def get_position_and_orientation(self):
        """
        获取车辆当前位置和朝向
        
        返回:
            tuple: (position, orientation) 位置是[x,y,z]，朝向是[roll,pitch,yaw]
        """
        pos, quat = p.getBasePositionAndOrientation(self.vehicle_id, physicsClientId=self.client_id)
        orientation = p.getEulerFromQuaternion(quat)
        return np.array(pos), np.array(orientation)
    
    def get_velocity(self):
        """
        获取车辆当前线速度
        
        返回:
            numpy.ndarray: [vx, vy, vz] 表示x,y,z方向的速度
        """
        linear_vel, _ = p.getBaseVelocity(self.vehicle_id, physicsClientId=self.client_id)
        return np.array(linear_vel)
    
    def get_angular_velocity(self):
        """
        获取车辆当前角速度
        
        返回:
            numpy.ndarray: [wx, wy, wz] 表示绕x,y,z轴的角速度
        """
        _, angular_vel = p.getBaseVelocity(self.vehicle_id, physicsClientId=self.client_id)
        return np.array(angular_vel)
    
    def check_collision(self):
        """
        检查车辆是否与其他物体发生碰撞
        
        返回:
            bool: 如果发生碰撞返回True，否则返回False
        """
        contact_points = p.getContactPoints(bodyA=self.vehicle_id, physicsClientId=self.client_id)
        return len(contact_points) > 0

    def perform_laser_scan(self, num_rays=16):
        """
        执行激光扫描，获取不同方向的距离
        
        参数:
            num_rays: 扫描射线数量
            
        返回:
            numpy.ndarray: 各个方向的距离值数组
        """
        try:
            # 获取当前位置和朝向
            pos, ori = self.get_position_and_orientation()
            position = np.array(pos)
            yaw = ori[2]
            
            # 创建旋转矩阵
            rotation_matrix = np.array([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            # 均匀分布的角度
            angles = np.linspace(-self.laser_angle_h, self.laser_angle_h, num_rays)
            
            # 存储各个方向的距离
            distances = np.ones(num_rays) * self.laser_max_distance
            
            # 清除旧的激光可视化线条
            for line in self.laser_lines:
                try:
                    p.removeUserDebugItem(line, physicsClientId=self.client_id)
                except:
                    pass  # 忽略删除调试项时的错误
            self.laser_lines = []
            
            # 对每个方向进行射线检测
            for i, angle in enumerate(angles):
                # 计算射线方向
                ray_orientation = np.array([
                    math.cos(angle),
                    math.sin(angle),
                    0
                ])
                
                # 应用车辆朝向
                ray_direction = rotation_matrix @ ray_orientation
                
                # 执行射线检测
                from_pos = position
                to_pos = [
                    position[0] + ray_direction[0] * self.laser_max_distance,
                    position[1] + ray_direction[1] * self.laser_max_distance,
                    position[2] + ray_direction[2] * self.laser_max_distance
                ]
                
                result = p.rayTest(from_pos, to_pos, physicsClientId=self.client_id)
                
                # 如果射线击中了物体
                if result[0][0] != -1:
                    hit_fraction = result[0][2]
                    distances[i] = hit_fraction * self.laser_max_distance
                    
                    # 可视化激光射线（击中的射线为红色）
                    hit_pos = result[0][3]
                    line_id = p.addUserDebugLine(
                        from_pos,
                        hit_pos,
                        lineColorRGB=[1, 0, 0],  # 红色
                        lineWidth=1.0,
                        lifeTime=1/30,  # 一帧的时间
                        physicsClientId=self.client_id
                    )
                    self.laser_lines.append(line_id)
                else:
                    # 可视化激光射线（未击中的射线为绿色）
                    line_id = p.addUserDebugLine(
                        from_pos,
                        to_pos,
                        lineColorRGB=[0, 1, 0],  # 绿色
                        lineWidth=1.0,
                        lifeTime=1/30,  # 一帧的时间
                        physicsClientId=self.client_id
                    )
                    self.laser_lines.append(line_id)
            
            return distances.astype(np.float32)
        
        except Exception as e:
            # 如果是连接断开错误，不再尝试执行其他操作
            if "Not connected to physics server" in str(e):
                raise  # 重新抛出错误，让调用者处理
            else:
                print(f"激光扫描过程中发生错误: {e}")
                # 返回默认值
                return np.ones(num_rays) * self.laser_max_distance

if __name__ == "__main__":
    # 测试代码
    import time
    from combat_arena import CombatArena
    
    # 创建竞技场
    arena = CombatArena(render_mode="gui", debug=True)
    
    # 创建两辆车
    vehicle1 = LaserVehicle(
        arena.client,
        start_pos=(-1.8, -1.8, 0.05),
        start_orientation=(0, 0, 0)
    )
    
    vehicle2 = LaserVehicle(
        arena.client,
        start_pos=(1.8, 1.8, 0.05),
        start_orientation=(0, 0, math.pi)
    )
    
    # 设置对手
    vehicle1.set_opponent(vehicle2.vehicle_id)
    vehicle2.set_opponent(vehicle1.vehicle_id)
    
    # 设置雷区信息
    vehicle1.set_mine_info(arena.get_mine_positions(), arena.get_mine_radius())
    vehicle2.set_mine_info(arena.get_mine_positions(), arena.get_mine_radius())
    
    # 模拟一段时间的控制
    try:
        step = 0
        while True:
            # 简单的控制策略：让车辆1绕圈行驶，车辆2原地旋转
            if step < 1000:
                vehicle1.apply_action([0.7, 0.5])  # 左转
                vehicle2.apply_action([0.5, -0.5])  # 原地旋转
            else:
                vehicle1.apply_action([0.5, 0.7])  # 右转
                vehicle2.apply_action([-0.5, 0.5])  # 反向旋转
            
            # 检查激光
            vehicle1.check_laser_hit()
            vehicle2.check_laser_hit()
            
            # 获取并打印状态信息（每100步）
            if step % 100 == 0:
                state1 = vehicle1.get_state()
                norm_state1 = vehicle1.normalize_state(state1)
                flat_state1 = vehicle1.get_flattened_state()
                
                print(f"Step {step}:")
                print(f"Position: {state1['self']['position']}")
                print(f"Laser hit time: {state1['opponent']['laser_hit_time']:.4f}s")
                print(f"Flattened state shape: {flat_state1.shape}")
                print("-" * 40)
            
            # 检查雷区碰撞
            arena.check_mine_collisions(vehicle1.vehicle_id)
            arena.check_mine_collisions(vehicle2.vehicle_id)
            
            # 步进模拟
            arena.step()
            step += 1
            
            time.sleep(1/240)
    
    except KeyboardInterrupt:
        arena.close()
        print("\n模拟已结束") 