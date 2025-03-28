import time
import math
import sys
import pybullet as p
from combat_arena import CombatArena

def main():
    print("创建战斗场景...")
    arena = None
    
    try:
        arena = CombatArena(render_mode="gui", debug=True)
        
        print("场景创建成功，请查看PyBullet窗口")
        print("添加车辆...")
        
        # 在左下角生成一个车辆
        vehicle_id = arena.spawn_vehicle(
            position=[-1.8, -1.8, 0.05], 
            orientation=[0, 0, math.pi/4],  # 朝向右上方
            size=(0.25, 0.2, 0.2)  # 使用15-30cm之间的尺寸
        )
        
        print("车辆添加成功")
        print("添加障碍物...")
        
        # 添加一些障碍物，用于激光测试
        arena.add_obstacle(
            position=[0, -1, 0.1],
            size=[0.3, 0.3, 0.2],
            color=[0.8, 0.2, 0.2, 1.0]  # 红色障碍物
        )
        
        arena.add_obstacle(
            position=[0, 1, 0.1],
            size=[0.3, 0.3, 0.2],
            color=[0.2, 0.2, 0.8, 1.0]  # 蓝色障碍物
        )
        
        print("障碍物添加成功")
        print("场景将在60秒后关闭，小车将开始移动并持续发射激光")
        
        # 让车辆移动60秒并展示激光
        start_time = time.time()
        connection_lost = False
        
        try:
            while time.time() - start_time < 60 and not connection_lost:
                if arena.vehicle:
                    # 让车辆做圆周运动，同时不断射出激光
                    t = time.time() - start_time
                    left_wheel = 0.5 + 0.3 * math.sin(t * 0.5)  # 左轮速度
                    right_wheel = 0.5 - 0.3 * math.sin(t * 0.5)  # 右轮速度
                    
                    try:
                        # 使用差速驱动控制：[左轮速度, 右轮速度]
                        arena.vehicle.apply_action([left_wheel, right_wheel])
                        
                        # 执行激光扫描
                        arena.vehicle.perform_laser_scan(num_rays=16)
                    except Exception as e:
                        if "Not connected to physics server" in str(e):
                            print("PyBullet连接已断开，无法继续控制车辆")
                            connection_lost = True
                            break
                        else:
                            print(f"应用动作时出错: {e}")
                
                try:
                    # 步进模拟
                    arena.step_simulation(1)
                except Exception as e:
                    if "Not connected to physics server" in str(e):
                        print("PyBullet连接已断开，模拟中止")
                        connection_lost = True
                        break
                    else:
                        print(f"步进模拟时出错: {e}")
                        break
                
                # 每0.01秒更新一次
                time.sleep(0.01)
                
                # 每5秒显示一次剩余时间
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and int(elapsed * 100) % 500 == 0:
                    print(f"剩余时间: {60-int(elapsed)}秒")
                    if arena.vehicle:
                        try:
                            pos, ori = arena.vehicle.get_position_and_orientation()
                            print(f"车辆位置: {pos}, 朝向: {ori}")
                        except Exception as e:
                            if "Not connected to physics server" in str(e):
                                print("PyBullet连接已断开，无法获取车辆位置")
                                connection_lost = True
                                break
                            else:
                                print(f"获取位置时出错: {e}")
            
            if connection_lost:
                print("由于PyBullet连接断开，演示提前结束")
            else:
                print("演示完成")
        
        except KeyboardInterrupt:
            print("用户中断演示")
    
    except Exception as e:
        print(f"初始化场景时发生错误: {e}")
    
    finally:
        print("正在关闭场景...")
        if arena:
            try:
                arena.close()
                print("场景已关闭")
            except Exception as e:
                print(f"关闭场景时发生错误: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        # 确保正确退出，即使发生错误
        sys.exit(1) 