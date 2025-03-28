"""
激光竞技场模拟器 - 测试脚本
用于测试战斗竞技场和激光车辆的各项功能
"""

import time
import math
import numpy as np
from combat_arena import CombatArena
from laser_vehicle import LaserVehicle

def test_arena():
    """测试竞技场功能"""
    print("开始测试竞技场...")
    
    # 创建竞技场
    arena = CombatArena(render_mode="gui", debug=True)
    
    # 添加一个简单的车辆
    vehicle_id = arena.spawn_vehicle(position=(-1.8, -1.8, 0.05), orientation=(0, 0, 0))
    
    # 注册碰撞回调
    def on_mine_collision(vehicle_id, mine_id):
        print(f"车辆 {vehicle_id} 与雷区 {mine_id} 发生碰撞!")
    
    arena.register_collision_callback(vehicle_id, on_mine_collision)
    
    # 运行一段时间模拟
    try:
        for i in range(500):
            # 每100帧输出一次信息
            if i % 100 == 0:
                print(f"模拟步数: {i}")
            
            # 检查碰撞
            arena.check_mine_collisions(vehicle_id)
            
            # 步进仿真
            arena.step()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass
    finally:
        arena.close()
        print("竞技场测试完成")

def test_vehicle():
    """测试车辆功能"""
    print("开始测试车辆...")
    
    # 创建竞技场
    arena = CombatArena(render_mode="gui", debug=True)
    
    # 创建车辆
    vehicle = LaserVehicle(
        arena.client,
        start_pos=(-1.8, -1.8, 0.05),
        start_orientation=(0, 0, 0)
    )
    
    # 设置雷区信息
    vehicle.set_mine_info(arena.get_mine_positions(), arena.get_mine_radius())
    
    # 测试差速驱动
    try:
        for i in range(1000):
            # 变化控制动作
            if i < 200:
                # 向前
                vehicle.apply_action([0.7, 0.7])
            elif i < 400:
                # 左转
                vehicle.apply_action([0.3, 0.7])
            elif i < 600:
                # 右转
                vehicle.apply_action([0.7, 0.3])
            elif i < 800:
                # 原地旋转
                vehicle.apply_action([-0.5, 0.5])
            else:
                # 后退
                vehicle.apply_action([-0.7, -0.7])
            
            # 每100帧输出一次状态
            if i % 100 == 0:
                state = vehicle.get_state()
                norm_state = vehicle.normalize_state(state)
                print(f"\n步数: {i}")
                print(f"位置: {state['self']['position']}")
                print(f"朝向: {state['self']['orientation']:.2f} rad")
                print(f"速度: {state['self']['linear_velocity']:.2f} m/s")
                print(f"角速度: {state['self']['angular_velocity']:.2f} rad/s")
                print(f"到最近雷区距离: {state['environment']['mine_distance']:.2f} m")
                print("-" * 40)
            
            # 绘制方向指示器
            vehicle.draw_direction_indicator()
            
            # 检查雷区碰撞
            arena.check_mine_collisions(vehicle.vehicle_id)
            
            # 步进仿真
            arena.step()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass
    finally:
        arena.close()
        print("车辆测试完成")

def test_state_normalization():
    """测试状态归一化功能"""
    print("开始测试状态归一化...")
    
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
    
    # 设置对手和雷区信息
    vehicle1.set_opponent(vehicle2.vehicle_id)
    vehicle2.set_opponent(vehicle1.vehicle_id)
    vehicle1.set_mine_info(arena.get_mine_positions(), arena.get_mine_radius())
    vehicle2.set_mine_info(arena.get_mine_positions(), arena.get_mine_radius())
    
    # 运行一些步骤让车辆移动
    for i in range(100):
        vehicle1.apply_action([0.5, 0.5])  # 直线前进
        vehicle2.apply_action([0.5, -0.5])  # 原地旋转
        
        # 绘制方向指示器
        vehicle1.draw_direction_indicator()
        vehicle2.draw_direction_indicator()
        
        arena.step()
    
    # 获取并打印状态信息
    state1 = vehicle1.get_state()
    norm_state1 = vehicle1.normalize_state(state1)
    flat_state1 = vehicle1.get_flattened_state()
    
    print("\n原始状态:")
    print(f"位置: {state1['self']['position']}")
    print(f"朝向: {state1['self']['orientation']:.2f} rad")
    print(f"线速度: {state1['self']['linear_velocity']:.2f} m/s")
    print(f"角速度: {state1['self']['angular_velocity']:.2f} rad/s")
    print(f"相对位置: {state1['opponent']['relative_position']}")
    print(f"相对速度: {state1['opponent']['relative_velocity']}")
    print(f"雷区距离: {state1['environment']['mine_distance']:.2f} m")
    
    print("\n归一化状态:")
    print(f"位置: {norm_state1['self']['position']}")
    print(f"朝向: {norm_state1['self']['orientation']:.2f}")
    print(f"线速度: {norm_state1['self']['linear_velocity']:.2f}")
    print(f"角速度: {norm_state1['self']['angular_velocity']:.2f}")
    print(f"相对位置: {norm_state1['opponent']['relative_position']}")
    print(f"相对速度: {norm_state1['opponent']['relative_velocity']}")
    print(f"雷区距离: {norm_state1['environment']['mine_distance']:.2f}")
    
    print(f"\n扁平化状态 (维度: {flat_state1.shape}):")
    print(flat_state1)
    
    # 验证是否在[-1,1]范围内
    min_val = flat_state1.min()
    max_val = flat_state1.max()
    print(f"\n归一化状态最小值: {min_val:.4f}, 最大值: {max_val:.4f}")
    print(f"是否在[-1,1]范围内: {min_val >= -1.0 and max_val <= 1.0}")
    
    arena.close()
    print("状态归一化测试完成")

if __name__ == "__main__":
    # 选择测试项
    print("请选择测试项目:")
    print("1. 测试竞技场")
    print("2. 测试车辆")
    print("3. 测试状态归一化")
    
    choice = input("请输入选择 (1-3): ")
    
    if choice == "1":
        test_arena()
    elif choice == "2":
        test_vehicle()
    elif choice == "3":
        test_state_normalization()
    else:
        print("无效选择！") 