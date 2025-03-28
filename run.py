#!/usr/bin/env python
"""
激光竞技场模拟器 - 启动脚本
提供多种运行方式选择
"""

import sys
import os
import argparse
from main import main
from test import test_arena, test_vehicle, test_state_normalization

def print_header():
    """打印程序标题"""
    header = """
    ==============================================
           激光车辆竞技场模拟器
    ==============================================
    """
    print(header)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="激光车辆竞技场模拟器")
    parser.add_argument("--mode", "-m", 
                        choices=["demo", "test_arena", "test_vehicle", "test_norm"],
                        default="demo",
                        help="运行模式: demo(演示), test_arena(测试竞技场), test_vehicle(测试车辆), test_norm(测试归一化)")
    
    return parser.parse_args()

def run_interactive():
    """交互式运行"""
    print_header()
    print("请选择运行模式:")
    print("1. 完整演示 (含控制面板)")
    print("2. 测试竞技场")
    print("3. 测试车辆")
    print("4. 测试状态归一化")
    print("0. 退出")
    
    choice = input("\n请输入选择 (0-4): ")
    
    if choice == "1":
        main()
    elif choice == "2":
        test_arena()
    elif choice == "3":
        test_vehicle()
    elif choice == "4":
        test_state_normalization()
    elif choice == "0":
        print("程序已退出")
        sys.exit(0)
    else:
        print("无效选择！")

def run_from_args(args):
    """根据命令行参数运行"""
    print_header()
    
    if args.mode == "demo":
        main()
    elif args.mode == "test_arena":
        test_arena()
    elif args.mode == "test_vehicle":
        test_vehicle()
    elif args.mode == "test_norm":
        test_state_normalization()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行参数模式
        args = parse_args()
        run_from_args(args)
    else:
        # 交互式模式
        run_interactive() 