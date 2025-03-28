import pybullet as p
import pybullet_data
import time
import numpy as np
import math
from collections import defaultdict

class CombatArena:
    """
    4x4米对抗场地，包含三个雷区和两个起始点
    坐标系：原点在场地中心，X轴向右，Y轴向上，Z轴垂直地面
    """
    
    def __init__(self, render_mode='direct', debug=False, client_id=None):
        """
        初始化战斗场景
        
        参数:
            render_mode: 渲染模式 ('direct', 'gui')
            debug: 是否打印调试信息
            client_id: PyBullet客户端ID，如果为None则创建新连接
        """
        self.debug = debug
        self.render_mode = render_mode
        
        # 使用传入的客户端ID或创建新连接
        if client_id is not None:
            self.client = client_id
            if self.debug:
                print(f"使用现有PyBullet客户端: {self.client}")
        else:
            try:
                if self.debug:
                    print(f"正在连接PyBullet引擎，模式: {render_mode}")
                
                if render_mode == 'gui':
                    self.client = p.connect(p.GUI)
                    # 设置GUI相机
                    p.resetDebugVisualizerCamera(
                        cameraDistance=5.0,
                        cameraYaw=0,
                        cameraPitch=-40,
                        cameraTargetPosition=[0, 0, 0]
                    )
                else:
                    self.client = p.connect(p.DIRECT)
                    
                if self.debug:
                    print(f"PyBullet引擎连接成功，客户端ID: {self.client}")
            except Exception as e:
                print(f"连接PyBullet引擎失败: {e}")
                raise
                
        # 初始化物理环境
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(1.0/240.0, physicsClientId=self.client)
        
        # 设置额外的搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        
        # 加载地面平面
        self.plane_id = p.loadURDF(
            "plane.urdf",
            [0, 0, 0],
            [0, 0, 0, 1],
            physicsClientId=self.client
        )
        
        # 存储场景中的对象
        self.vehicle = None
        self.obstacles = []
        self.targets = []
        self.mine_positions = []
        
        # 初始化随机数生成器
        self.rng = np.random.RandomState(int(time.time()))
        
        # 竞技场参数
        self.arena_size = 4.0  # 4x4米场地
        self.arena_half_size = self.arena_size / 2
        self.mine_radius = 0.5  # 雷区半径
        self.mine_height = 0.001  # 雷区高度改为1mm，贴地
        
        # 雷区中心位置 - 修正为紧贴排列
        # 每个雷区直径1米，中心间距应为1米
        self.mine_positions = [
            (-1.0, 0, self.mine_height/2),  # 左雷区
            (0, 0, self.mine_height/2),     # 中雷区
            (1.0, 0, self.mine_height/2)    # 右雷区
        ]
        
        # 存储物体列表
        self.mines = []        # 雷区（黑色圆环）
        self.mine_objects = [] # 地雷（黄色圆形）
        self.boundaries = []
        self.debug_lines = []
        self.start_markers = []
        
        # 车辆起始位置 - 修改为左下角和右下角
        self.start_positions = [
            [-self.arena_half_size + 0.5, -self.arena_half_size + 0.5, 0.1],  # 左下角
            [self.arena_half_size - 0.5, self.arena_half_size + 0.5, 0.1]     # 右下角（已修改）
        ]
        
        # 设置随机种子
        np.random.seed(int(time.time()))
        
        # 碰撞检测
        self.collision_count = defaultdict(int)
        
        # 创建默认车辆
        self.opponent = None
        
        # 初始化场地
        try:
            self._create_ground()
            self._create_boundaries()
            self._create_mines()
            
            if self.debug:
                self._create_grid()
                self._create_start_markers()
        except Exception as e:
            print(f"初始化场地失败: {e}")
            p.disconnect(self.client)
            raise
    
    def _create_ground(self):
        """创建地面"""
        self.ground_id = p.createCollisionShape(p.GEOM_PLANE)
        self.ground = p.createMultiBody(0, self.ground_id)
        
        # 设置地面颜色为浅灰色
        p.changeVisualShape(self.ground, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
    
    def _create_boundaries(self):
        """创建场地边界和贴地黑色标志线"""
        self.boundaries = []
        
        # 创建4*4米场地的黑色标志线（贴地）- 增加线宽
        line_width = 0.05  # 线宽5cm，原先是2cm
        line_height = 0.001  # 线高1mm，贴地
        
        # 添加四条黑色标志线
        # 下边界线
        line_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.arena_half_size, line_width/2, line_height/2])
        line = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=line_shape,
            basePosition=[0, -self.arena_half_size, line_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        p.changeVisualShape(line, -1, rgbaColor=[0.0, 0.0, 0.0, 1.0])  # 纯黑色
        self.boundaries.append(line)
        
        # 上边界线
        line = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=line_shape,
            basePosition=[0, self.arena_half_size, line_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        p.changeVisualShape(line, -1, rgbaColor=[0.0, 0.0, 0.0, 1.0])  # 纯黑色
        self.boundaries.append(line)
        
        # 左边界线
        line_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[line_width/2, self.arena_half_size, line_height/2])
        line = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=line_shape,
            basePosition=[-self.arena_half_size, 0, line_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        p.changeVisualShape(line, -1, rgbaColor=[0.0, 0.0, 0.0, 1.0])  # 纯黑色
        self.boundaries.append(line)
        
        # 右边界线
        line = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=line_shape,
            basePosition=[self.arena_half_size, 0, line_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        p.changeVisualShape(line, -1, rgbaColor=[0.0, 0.0, 0.0, 1.0])  # 纯黑色
        self.boundaries.append(line)
        
        # 创建隐形物理边界墙（防止车辆冲出场地，但不可见）
        wall_thickness = 0.1  # 厚度
        wall_height = 0.3     # 高度
        
        # 下边界 (y = -2)
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.arena_half_size, wall_thickness/2, wall_height/2])
        wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            basePosition=[0, -self.arena_half_size, wall_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        # 设置为完全透明
        p.changeVisualShape(wall, -1, rgbaColor=[0, 0, 0, 0])
        self.boundaries.append(wall)
        
        # 上边界 (y = 2)
        wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            basePosition=[0, self.arena_half_size, wall_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        # 设置为完全透明
        p.changeVisualShape(wall, -1, rgbaColor=[0, 0, 0, 0])
        self.boundaries.append(wall)
        
        # 左边界 (x = -2)
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, self.arena_half_size, wall_height/2])
        wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            basePosition=[-self.arena_half_size, 0, wall_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        # 设置为完全透明
        p.changeVisualShape(wall, -1, rgbaColor=[0, 0, 0, 0])
        self.boundaries.append(wall)
        
        # 右边界 (x = 2)
        wall = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_shape,
            basePosition=[self.arena_half_size, 0, wall_height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        # 设置为完全透明
        p.changeVisualShape(wall, -1, rgbaColor=[0, 0, 0, 0])
        self.boundaries.append(wall)
    
    def _create_mines(self):
        """创建雷区（黑色圆环）和地雷（黄色圆形）"""
        self.mines = []
        self.mine_objects = []  # 存储地雷对象
        
        # 创建雷区的内外圆半径参数
        outer_radius = self.mine_radius  # 外圆半径0.5米
        inner_radius = outer_radius - 0.05  # 内圆半径0.45米，形成5cm宽的圆环
        
        for pos in self.mine_positions:
            # 创建贴地黑色圆环（雷区）
            # 首先，创建一个完美的黑色圆环，使用PyBullet的可视化形状
            
            # 外圆（黑色圆盘）
            outer_visual = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=outer_radius,
                length=self.mine_height,
                rgbaColor=[0.0, 0.0, 0.0, 1.0]  # 纯黑色
            )
            
            # 内圆（与地面同色，用于"挖空"）
            inner_visual = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=inner_radius,
                length=self.mine_height * 1.1,  # 稍微高一点，确保覆盖
                rgbaColor=[0.9, 0.9, 0.9, 1.0]  # 与地面同色
            )
            
            # 外圆物理模型（用于碰撞检测）
            outer_collision = p.createCollisionShape(
                p.GEOM_CYLINDER, 
                radius=outer_radius,
                height=self.mine_height
            )
            
            # 创建外圆多体
            outer_mine = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=outer_collision,
                baseVisualShapeIndex=outer_visual,
                basePosition=[pos[0], pos[1], self.mine_height/2]  # 贴地
            )
            self.mines.append(outer_mine)
            
            # 创建内圆多体（无碰撞形状，仅用于视觉效果）
            inner_mine = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,  # 无碰撞形状
                baseVisualShapeIndex=inner_visual,
                basePosition=[pos[0], pos[1], self.mine_height/2 * 1.1]  # 贴地且略高
            )
            
            # 创建黄色地雷（直径15cm的黄色圆形）
            mine_visual = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=0.075,  # 直径15cm，半径7.5cm
                length=0.02,  # 高度2cm
                rgbaColor=[1.0, 0.8, 0.0, 1.0]  # 黄色
            )
            
            mine_collision = p.createCollisionShape(
                p.GEOM_CYLINDER, 
                radius=0.075,
                height=0.02
            )
            
            mine_object = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=mine_collision,
                baseVisualShapeIndex=mine_visual,
                basePosition=[pos[0], pos[1], 0.01]  # 放在雷区中心上方
            )
            self.mine_objects.append(mine_object)
    
    def add_mine(self, position, radius=None):
        """添加一个新的雷区
        
        参数:
            position: 雷区中心位置 [x, y, z]
            radius: 雷区半径（可选，默认使用环境默认值）
        
        返回:
            int: 创建的雷区物理ID
        """
        if radius is None:
            radius = self.mine_radius
        
        # 创建雷区的内外圆半径参数
        outer_radius = radius  # 外圆半径
        inner_radius = outer_radius - 0.05  # 内圆半径，使其形成5cm宽的圆环
        
        # 外圆（黑色圆盘）
        outer_visual = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=outer_radius,
            length=self.mine_height,
            rgbaColor=[0.0, 0.0, 0.0, 1.0]  # 纯黑色
        )
        
        # 内圆（与地面同色，用于"挖空"）
        inner_visual = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=inner_radius,
            length=self.mine_height * 1.1,  # 稍微高一点，确保覆盖
            rgbaColor=[0.9, 0.9, 0.9, 1.0]  # 与地面同色
        )
        
        # 外圆物理模型（用于碰撞检测）
        outer_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=outer_radius,
            height=self.mine_height
        )
        
        # 创建外圆多体
        outer_mine = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=outer_collision,
            baseVisualShapeIndex=outer_visual,
            basePosition=[position[0], position[1], position[2]]  # 贴地
        )
        self.mines.append(outer_mine)
        
        # 创建内圆多体（无碰撞形状，仅用于视觉效果）
        inner_mine = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # 无碰撞形状
            baseVisualShapeIndex=inner_visual,
            basePosition=[position[0], position[1], position[2] * 1.1]  # 贴地且略高
        )
        
        # 创建黄色地雷（直径15cm的黄色圆形）
        mine_visual = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=0.075,  # 直径15cm，半径7.5cm
            length=0.02,  # 高度2cm
            rgbaColor=[1.0, 0.8, 0.0, 1.0]  # 黄色
        )
        
        mine_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=0.075,
            height=0.02
        )
        
        mine_object = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=mine_collision,
            baseVisualShapeIndex=mine_visual,
            basePosition=[position[0], position[1], 0.01]  # 放在雷区中心上方
        )
        self.mine_objects.append(mine_object)
        
        self.mine_positions.append(position)
        return outer_mine
    
    def add_obstacle(self, position, size, mass=0, color=None):
        """添加一个障碍物
        
        参数:
            position: 障碍物中心位置 [x, y, z]
            size: 障碍物尺寸 [length, width, height]
            mass: 障碍物质量，0表示静态障碍物
            color: RGBA颜色，默认为灰色
            
        返回:
            int: 创建的障碍物物理ID
        """
        if color is None:
            color = [0.5, 0.5, 0.5, 1]  # 默认灰色
            
        half_extents = [s/2 for s in size]
        obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        
        obstacle = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=obstacle_shape,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=position
        )
        
        self.obstacles.append(obstacle)
        return obstacle
    
    def _create_grid(self):
        """创建网格线用于可视化"""
        self.debug_lines = []
        
        # 网格线参数
        grid_size = 0.5  # 网格大小
        line_width = 1.0  # 线宽
        line_color = [0.7, 0.7, 0.7]  # 灰色
        line_height = 0.001  # 线距地面高度
        
        # 创建水平和垂直线
        for i in range(-int(self.arena_half_size / grid_size), int(self.arena_half_size / grid_size) + 1):
            # 计算线的位置
            pos = i * grid_size
            
            # 水平线 (固定 x)
            line_from = [pos, -self.arena_half_size, line_height]
            line_to = [pos, self.arena_half_size, line_height]
            line_id = p.addUserDebugLine(line_from, line_to, line_color, line_width)
            self.debug_lines.append(line_id)
            
            # 垂直线 (固定 y)
            line_from = [-self.arena_half_size, pos, line_height]
            line_to = [self.arena_half_size, pos, line_height]
            line_id = p.addUserDebugLine(line_from, line_to, line_color, line_width)
            self.debug_lines.append(line_id)
        
        # 添加坐标轴方向指示线
        # X轴 (红色)
        p.addUserDebugLine([0, 0, 0.01], [0.5, 0, 0.01], [1, 0, 0], 3)
        # Y轴 (绿色)
        p.addUserDebugLine([0, 0, 0.01], [0, 0.5, 0.01], [0, 1, 0], 3)
        # Z轴 (蓝色)
        p.addUserDebugLine([0, 0, 0.01], [0, 0, 0.5], [0, 0, 1], 3)
    
    def _create_start_markers(self):
        """创建起始位置标记"""
        # 车辆尺寸为0.2m x 0.15m，标记尺寸比车大一半
        marker_size_x = 0.2 * 1.5  # 车长的1.5倍
        marker_size_y = 0.15 * 1.5  # 车宽的1.5倍
        marker_height = 0.001  # 贴地，1mm高度
        
        # 修改出生点位置 - 紧贴场景边缘
        # 左下角出生点位置
        left_bottom_pos = [-self.arena_half_size + marker_size_x/2, -self.arena_half_size + marker_size_y/2, 0.001]
        # 右上角出生点位置
        right_top_pos = [self.arena_half_size - marker_size_x/2, self.arena_half_size - marker_size_y/2, 0.001]
        
        # 更新出生点位置
        self.start_positions = [left_bottom_pos, right_top_pos]
        
        # 己方起始点（蓝色）- 使用更鲜艳的蓝色
        start_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[marker_size_x/2, marker_size_y/2, marker_height/2],
            rgbaColor=[0.0, 0.5, 1.0, 0.8]  # 更鲜艳的蓝色，稍微更不透明
        )
        marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_visual,
            basePosition=left_bottom_pos  # 左下角，紧贴边缘
        )
        self.start_markers.append(marker)
        
        # 对方起始点（红色）- 使用更鲜艳的红色
        opponent_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[marker_size_x/2, marker_size_y/2, marker_height/2],
            rgbaColor=[1.0, 0.3, 0.0, 0.8]  # 更鲜艳的红色，稍微更不透明
        )
        marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=opponent_visual,
            basePosition=right_top_pos  # 右上角，紧贴边缘
        )
        self.start_markers.append(marker)
        
        # 添加文字标签
        p.addUserDebugText(
            text="蓝方出发点",
            textPosition=[left_bottom_pos[0], left_bottom_pos[1] - 0.3, 0.05],
            textColorRGB=[0, 0.5, 1],
            textSize=1.2
        )
        
        p.addUserDebugText(
            text="红方出发点",
            textPosition=[right_top_pos[0], right_top_pos[1] - 0.3, 0.05],
            textColorRGB=[1, 0.3, 0],
            textSize=1.2
        )
    
    def register_collision_callback(self, body_id, callback_func):
        """
        注册碰撞回调函数
        
        参数:
            body_id: 要监测碰撞的物体ID
            callback_func: 碰撞发生时调用的函数，接收参数(body_id, collided_with_id)
        """
        self.collision_callbacks[body_id].append(callback_func)
    
    def check_mine_collisions(self, vehicle_id):
        """
        检查车辆是否与雷区碰撞
        
        参数:
            vehicle_id: 车辆的物理ID
            
        返回:
            bool: 是否检测到碰撞
        """
        # 获取车辆与所有雷区的接触点
        for mine_id in self.mines:
            points = p.getContactPoints(vehicle_id, mine_id)
            if len(points) > 0:
                # 执行已注册的回调函数
                for callback in self.collision_callbacks.get(vehicle_id, []):
                    callback(vehicle_id, mine_id)
                
                # 增加碰撞计数
                self.collision_count[vehicle_id] += 1
                return True
        return False
    
    def spawn_vehicle(self, position, orientation=(0, 0, 0), mass=1.5, size=(0.25, 0.2, 0.2)):
        """
        在场地中生成一个简单的车辆模型
        
        参数:
            position: (x, y, z) 初始位置
            orientation: (roll, pitch, yaw) 初始朝向（弧度）
            mass: 车辆质量（kg）
            size: (length, width, height) 车辆尺寸（米）
            
        返回:
            int: 生成的车辆ID
        """
        from laser_vehicle import LaserVehicle
        # 创建激光车辆实例 - 更新尺寸在15-30cm之间
        vehicle = LaserVehicle(
            client_id=self.client,
            start_pos=position,
            start_orientation=orientation,
            vehicle_size=size,  # 默认值改为(0.25, 0.2, 0.2)，满足15-30cm要求
            mass=mass,
            color=[0.1, 0.1, 0.8, 1]  # 蓝色
        )
        
        # 保存为主车辆
        if self.vehicle is None:
            self.vehicle = vehicle
            
        return vehicle.vehicle_id
    
    def step_simulation(self, steps=1):
        """执行多步模拟
        
        参数:
            steps: 执行步数
        """
        try:
            for _ in range(steps):
                p.stepSimulation(physicsClientId=self.client)
                
                # 如果有车辆，检查与雷区的碰撞
                if self.vehicle and hasattr(self.vehicle, 'vehicle_id'):
                    self.check_mine_collisions(self.vehicle.vehicle_id)
        except p.error as e:
            # 处理PyBullet连接错误
            if "Not connected to physics server" in str(e):
                print("PyBullet连接已断开，尝试重新连接...")
                try:
                    # 尝试重新连接
                    if self.render_mode == 'gui':
                        self.client = p.connect(p.GUI)
                    else:
                        self.client = p.connect(p.DIRECT)
                    
                    print(f"已重新连接到PyBullet，客户端ID: {self.client}")
                    # 重置物理环境
                    p.setGravity(0, 0, -9.8, physicsClientId=self.client)
                    p.setRealTimeSimulation(0, physicsClientId=self.client)
                    p.setTimeStep(1.0/240.0, physicsClientId=self.client)
                    
                    # 如果重连成功但场景状态已丢失，可能需要重新创建场景
                    print("警告：物理服务器连接已恢复，但场景状态可能已丢失。建议重新启动模拟。")
                except Exception as reconnect_error:
                    print(f"重新连接失败: {reconnect_error}")
                    raise
            else:
                print(f"步进模拟时发生错误: {e}")
                raise
    
    def step(self):
        """执行一步模拟"""
        self.step_simulation(1)
    
    def close(self):
        """关闭物理引擎连接"""
        try:
            # 检查连接是否仍然有效
            p.getConnectionInfo(physicsClientId=self.client)
            # 如果连接有效，则断开连接
            p.disconnect(self.client)
            print("成功关闭PyBullet连接")
        except Exception as e:
            print(f"关闭PyBullet连接时发生错误: {e}")
            # 连接可能已经断开，忽略错误
    
    def reset(self):
        """重置场地状态"""
        # 清除当前所有物体并重新创建场地
        p.resetSimulation(physicsClientId=self.client)
        
        # 重置内部变量
        self.mines = []
        self.mine_objects = []  # 重置地雷对象列表
        self.obstacles = []
        self.vehicle = None
        self.opponent = None
        self.boundaries = []
        self.debug_lines = []
        self.start_markers = []
        
        self.mine_positions = [
            (-1.0, 0, self.mine_height/2),  # 左雷区
            (0, 0, self.mine_height/2),     # 中雷区
            (1.0, 0, self.mine_height/2)    # 右雷区
        ]
        
        # 重新创建场地
        self._create_ground()
        self._create_boundaries()
        self._create_mines()
        
        # 如果是debug模式，重新创建辅助网格
        if self.debug:
            self._create_grid()
            self._create_start_markers()
        
        # 重置碰撞记录
        self.collision_callbacks = defaultdict(list)
        self.collision_count = defaultdict(int)
    
    def get_mine_positions(self):
        """获取所有雷区的位置"""
        return self.mine_positions
    
    def get_mine_radius(self):
        """获取雷区半径"""
        return self.mine_radius

if __name__ == "__main__":
    # 测试代码
    arena = CombatArena(render_mode="gui", debug=True)
    
    # 添加障碍物
    arena.add_obstacle([0, 1, 0.25], [0.5, 0.5, 0.5], color=[0.8, 0.2, 0.2, 1])
    
    # 创建车辆
    vehicle_id = arena.spawn_vehicle(position=(-1.8, -1.8, 0.05), orientation=(0, 0, 0))
    
    # 注册碰撞回调函数
    def on_mine_collision(vehicle_id, mine_id):
        print(f"车辆 {vehicle_id} 与雷区 {mine_id} 发生碰撞!")
    
    arena.register_collision_callback(vehicle_id, on_mine_collision)
    
    # 运行模拟
    try:
        while True:
            arena.step()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass
    
    arena.close() 