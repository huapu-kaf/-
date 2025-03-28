import pybullet as p
import pybullet_data
import time
import numpy as np

def main():
    """
    简单的PyBullet GUI测试
    """
    print("启动简单的PyBullet GUI测试...")
    
    # 连接到GUI模式
    physicsClient = p.connect(p.GUI)
    if physicsClient < 0:
        print("无法连接到PyBullet GUI，请检查您的图形驱动和PyBullet安装")
        return
    
    print(f"成功连接到PyBullet，客户端ID: {physicsClient}")
    
    # 设置额外的搜索路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置重力
    p.setGravity(0, 0, -9.81)
    
    # 加载地面
    planeId = p.loadURDF("plane.urdf")
    print(f"加载地面，ID: {planeId}")
    
    # 加载一个简单的物体
    cubeStartPos = [0, 0, 1]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
    print(f"加载R2D2模型，ID: {boxId}")
    
    # 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # 添加一些调试文本
    textId = p.addUserDebugText(
        text="PyBullet GUI测试 - 按ESC退出",
        textPosition=[0, 0, 2],
        textColorRGB=[1, 0, 0],
        textSize=1.5
    )
    
    # 添加一个调试参数滑块
    sliderA = p.addUserDebugParameter("测试参数", 0, 10, 5)
    
    print("GUI已启动，请检查是否出现窗口。按Ctrl+C结束测试。")
    
    try:
        # 主循环
        for i in range(10000):
            # 读取滑块值
            paramValue = p.readUserDebugParameter(sliderA)
            
            # 更新文本
            p.addUserDebugText(
                text=f"PyBullet GUI测试 - 参数值: {paramValue:.2f}",
                textPosition=[0, 0, 2],
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                replaceItemUniqueId=textId
            )
            
            # 模拟一步
            p.stepSimulation()
            
            # 暂停一下，让GUI有时间响应
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("用户中断测试")
    finally:
        # 断开连接
        p.disconnect()
        print("测试结束，已断开与PyBullet的连接")

if __name__ == "__main__":
    main() 