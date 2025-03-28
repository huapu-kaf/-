"""
激光竞技场模拟器 - 主程序
用于演示和测试战斗竞技场和激光车辆功能
"""

import time
import math
import numpy as np
import pybullet as p
from combat_arena import CombatArena
from laser_vehicle import LaserVehicle

def main():
    """主函数"""
    # 创建战斗竞技场
    print("初始化战斗竞技场...")
    arena = CombatArena(render_mode="gui", debug=True)
    
    # 创建两辆激光车辆
    print("创建激光车辆...")
    vehicle1 = LaserVehicle(
        arena.client,
        start_pos=(-1.8, -1.8, 0.05),
        start_orientation=(0, 0, 0),
        color=[0, 0.4, 1, 1]  # 蓝色
    )
    
    vehicle2 = LaserVehicle(
        arena.client,
        start_pos=(1.8, 1.8, 0.05),
        start_orientation=(0, 0, math.pi),
        color=[1, 0.4, 0, 1]  # 红色
    )
    
    # 设置对手关系
    vehicle1.set_opponent(vehicle2.vehicle_id)
    vehicle2.set_opponent(vehicle1.vehicle_id)
    
    # 设置雷区信息
    mine_positions = arena.get_mine_positions()
    mine_radius = arena.get_mine_radius()
    vehicle1.set_mine_info(mine_positions, mine_radius)
    vehicle2.set_mine_info(mine_positions, mine_radius)
    
    # 注册碰撞回调
    def on_mine_collision(vehicle_id, mine_id):
        if vehicle_id == vehicle1.vehicle_id:
            print(f"车辆1 与雷区 {mine_id} 发生碰撞!")
        else:
            print(f"车辆2 与雷区 {mine_id} 发生碰撞!")
    
    arena.register_collision_callback(vehicle1.vehicle_id, on_mine_collision)
    arena.register_collision_callback(vehicle2.vehicle_id, on_mine_collision)
    
    # 添加调试信息
    p.addUserDebugText(
        text="车辆1",
        textPosition=[-1.8, -1.8, 0.2],
        textColorRGB=[0, 0, 1],
        textSize=1.5
    )
    
    p.addUserDebugText(
        text="车辆2",
        textPosition=[1.8, 1.8, 0.2],
        textColorRGB=[0, 0, 1],
        textSize=1.5
    )
    
    # 添加状态显示
    state_display_1 = p.addUserDebugText(
        text="",
        textPosition=[-1.95, -1.95, 0.5],
        textColorRGB=[1, 1, 1],
        textSize=1.0
    )
    
    state_display_2 = p.addUserDebugText(
        text="",
        textPosition=[0.5, 0.5, 0.5],
        textColorRGB=[1, 1, 1],
        textSize=1.0
    )
    
    # 创建演示控制面板
    print("创建控制面板...")
    sliders = []
    
    # 添加一个标题和参数面板
    p.addUserDebugText(
        text="激光车辆对抗竞技场",
        textPosition=[0, 0, 1.0],
        textColorRGB=[0.2, 0.6, 1.0],
        textSize=2.0
    )
    
    # 添加参数显示面板
    param_display = p.addUserDebugText(
        text=(
            "─────── 模拟参数 ───────\n"
            f"场地大小: 4x4米\n"
            f"雷区直径: 1米 (共3个)\n"
            f"车辆尺寸: 20x15x10厘米\n"
            f"最大速度: {vehicle1.max_velocity:.1f}米/秒\n"
            f"最大角速度: {vehicle1.max_angular_velocity/math.pi:.1f}π弧度/秒\n"
            f"加速度限制: {vehicle1.max_acceleration:.1f}米/秒²\n"
            f"激光角度: 水平±60°，垂直±15°\n"
            "──────────────────"
        ),
        textPosition=[-1.95, 0, 1.0],
        textColorRGB=[1.0, 1.0, 0.8],
        textSize=1.2
    )
    
    # 车辆1控制滑块
    vehicle1_left_wheel = p.addUserDebugParameter("车辆1左轮速度", -1.0, 1.0, 0.0)
    vehicle1_right_wheel = p.addUserDebugParameter("车辆1右轮速度", -1.0, 1.0, 0.0)
    sliders.append(vehicle1_left_wheel)
    sliders.append(vehicle1_right_wheel)
    
    # 车辆2控制滑块
    vehicle2_left_wheel = p.addUserDebugParameter("车辆2左轮速度", -1.0, 1.0, 0.0)
    vehicle2_right_wheel = p.addUserDebugParameter("车辆2右轮速度", -1.0, 1.0, 0.0)
    sliders.append(vehicle2_left_wheel)
    sliders.append(vehicle2_right_wheel)
    
    # 自动演示按钮
    demo_mode = p.addUserDebugParameter("自动演示模式", 0, 1, 0)
    sliders.append(demo_mode)
    
    # 重置按钮
    reset_button = p.addUserDebugParameter("重置场景", 1, 0, 0)
    last_reset_value = p.readUserDebugParameter(reset_button)
    sliders.append(reset_button)
    
    print("开始模拟...")
    
    # 自动演示模式的轨迹
    demo_time = 0
    
    # 主循环
    try:
        while True:
            # 检查重置按钮
            reset_value = p.readUserDebugParameter(reset_button)
            if reset_value != last_reset_value:
                last_reset_value = reset_value
                
                # 关闭旧的竞技场
                arena.close()
                
                # 重新创建场景
                arena = CombatArena(render_mode="gui", debug=True)
                
                # 重新创建车辆
                vehicle1 = LaserVehicle(
                    arena.client,
                    start_pos=(-1.8, -1.8, 0.05),
                    start_orientation=(0, 0, 0),
                    color=[0, 0.4, 1, 1]  # 蓝色
                )
                
                vehicle2 = LaserVehicle(
                    arena.client,
                    start_pos=(1.8, 1.8, 0.05),
                    start_orientation=(0, 0, math.pi),
                    color=[1, 0.4, 0, 1]  # 红色
                )
                
                # 重新设置车辆关系
                vehicle1.set_opponent(vehicle2.vehicle_id)
                vehicle2.set_opponent(vehicle1.vehicle_id)
                
                # 重新设置雷区信息
                mine_positions = arena.get_mine_positions()
                mine_radius = arena.get_mine_radius()
                vehicle1.set_mine_info(mine_positions, mine_radius)
                vehicle2.set_mine_info(mine_positions, mine_radius)
                
                # 重新注册回调
                arena.register_collision_callback(vehicle1.vehicle_id, on_mine_collision)
                arena.register_collision_callback(vehicle2.vehicle_id, on_mine_collision)
                
                # 重置演示时间
                demo_time = 0
                
                print("场景已重置")
            
            # 检查是否处于自动演示模式
            is_demo_mode = p.readUserDebugParameter(demo_mode) > 0.5
            
            if is_demo_mode:
                # 自动演示模式 - 预设的轨迹
                demo_time += 1.0 / 240
                
                # 车辆1: 8字形轨迹
                t1 = demo_time * 0.5
                if (t1 % 10) < 5:
                    # 左圈
                    vehicle1.apply_action([0.7, 0.3])
                else:
                    # 右圈
                    vehicle1.apply_action([0.3, 0.7])
                
                # 车辆2: 靠近车辆1
                state1 = vehicle1.get_state()
                state2 = vehicle2.get_state()
                
                # 计算朝向车辆1的方向
                dx = state1['self']['position'][0] - state2['self']['position'][0]
                dy = state1['self']['position'][1] - state2['self']['position'][1]
                angle_to_vehicle1 = math.atan2(dy, dx)
                
                # 获取当前朝向
                current_angle = state2['self']['orientation']
                
                # 计算角度差
                angle_diff = angle_to_vehicle1 - current_angle
                # 归一化到 [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # 根据角度差调整左右轮速度
                if abs(angle_diff) < 0.1:
                    # 直行
                    vehicle2.apply_action([0.7, 0.7])
                elif angle_diff > 0:
                    # 左转
                    vehicle2.apply_action([0.2, 0.8])
                else:
                    # 右转
                    vehicle2.apply_action([0.8, 0.2])
            else:
                # 手动控制模式 - 使用滑块值
                v1_left = p.readUserDebugParameter(vehicle1_left_wheel)
                v1_right = p.readUserDebugParameter(vehicle1_right_wheel)
                v2_left = p.readUserDebugParameter(vehicle2_left_wheel)
                v2_right = p.readUserDebugParameter(vehicle2_right_wheel)
                
                # 应用到车辆
                vehicle1.apply_action([v1_left, v1_right])
                vehicle2.apply_action([v2_left, v2_right])
            
            # 检查激光命中
            vehicle1_hit = vehicle1.check_laser_hit()
            vehicle2_hit = vehicle2.check_laser_hit()
            
            # 检查雷区碰撞
            arena.check_mine_collisions(vehicle1.vehicle_id)
            arena.check_mine_collisions(vehicle2.vehicle_id)
            
            # 更新状态显示
            state1 = vehicle1.get_state()
            state2 = vehicle2.get_state()
            
            # 绘制方向指示器
            vehicle1.draw_direction_indicator()
            vehicle2.draw_direction_indicator()
            
            # 创建状态显示文本
            state_text_1 = (
                f"┌─── 车辆1状态 ───┐\n"
                f"│ 位置: ({state1['self']['position'][0]:.2f}, {state1['self']['position'][1]:.2f}) │\n"
                f"│ 朝向: {state1['self']['orientation']*180/math.pi:.1f}° │\n"
                f"│ 速度: {state1['self']['linear_velocity']:.2f} m/s │\n"
                f"│ 被照射: {state1['opponent']['laser_hit_time']*1000:.1f} ms │\n"
                f"│ 雷区距离: {state1['environment']['mine_distance']:.2f} m │\n"
                f"└────────────┘"
            )
            
            state_text_2 = (
                f"┌─── 车辆2状态 ───┐\n"
                f"│ 位置: ({state2['self']['position'][0]:.2f}, {state2['self']['position'][1]:.2f}) │\n"
                f"│ 朝向: {state2['self']['orientation']*180/math.pi:.1f}° │\n"
                f"│ 速度: {state2['self']['linear_velocity']:.2f} m/s │\n"
                f"│ 被照射: {state2['opponent']['laser_hit_time']*1000:.1f} ms │\n"
                f"│ 雷区距离: {state2['environment']['mine_distance']:.2f} m │\n"
                f"└────────────┘"
            )
            
            # 更新文本显示
            p.addUserDebugText(
                text=state_text_1,
                textPosition=[-1.95, -1.95, 0.5],
                textColorRGB=[1, 1, 1],
                textSize=1.0,
                replaceItemUniqueId=state_display_1
            )
            
            p.addUserDebugText(
                text=state_text_2,
                textPosition=[0.5, 0.5, 0.5],
                textColorRGB=[1, 1, 1],
                textSize=1.0,
                replaceItemUniqueId=state_display_2
            )
            
            # 步进仿真
            arena.step()
            time.sleep(1/240)  # 控制帧率
            
    except KeyboardInterrupt:
        print("\n模拟已结束")
        arena.close()

if __name__ == "__main__":
    main()
