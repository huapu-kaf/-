"""
测试PyBullet的GUI功能
"""

import pybullet as p
import pybullet_data
import time
import numpy as np

print("启动PyBullet GUI测试...")
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 加载地面
plane_id = p.loadURDF("plane.urdf")

# 加载一个机器人
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

# 添加一些调试文本
p.addUserDebugText(
    text="PyBullet GUI测试",
    textPosition=[0, 0, 1],
    textColorRGB=[1, 0, 0],
    textSize=1.5
)

print("GUI已启动，请检查是否出现窗口。按Ctrl+C结束测试。")
try:
    while True:
        p.stepSimulation()
        time.sleep(1/240.0)
except KeyboardInterrupt:
    p.disconnect()
    print("测试结束。") 