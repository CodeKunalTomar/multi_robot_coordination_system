#!/usr/bin/env python3
"""
Multi-Robot Tic-Tac-Toe: Dual Arm Environment with Five Virtual Cameras
Program 1: Foundation - Dual arm setup with comprehensive camera monitoring
"""

import os
import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np

class DualArmEnvironment:
    def __init__(self):
        """Initialize the dual-arm tic-tac-toe environment"""
        # Setup working directory
        self.working_dir = os.path.expanduser("~/multi_robot_tictactoe")
        os.makedirs(self.working_dir, exist_ok=True)
        os.chdir(self.working_dir)
        print(f"🤖 Working directory: {os.getcwd()}")
        
        # Initialize PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Environment objects
        self.plane_id = None
        self.table_id = None
        self.board_id = None
        
        # Robot arms
        self.robot_arm1_id = None  # Player X (North side)
        self.robot_arm2_id = None  # Player O (South side)
        
        # Camera parameters
        self.camera_configs = self._setup_camera_configs()
        self.camera_windows = {}
        
        # Initialize the environment
        self._setup_environment()
        self._setup_robots()
        self._initialize_camera_windows()
        
        print("✅ Dual-arm environment initialized successfully!")
    
    def _setup_camera_configs(self):
        """Configure five virtual cameras around the table"""
        return {
            "camera_1_north": {
                "position": [0, -1.5, 1.0],
                "target": [0, 0.6, 0.65],
                "name": "North Side View (Arm 1 Perspective)"
            },
            "camera_2_east": {
                "position": [1.5, 0.6, 1.0],
                "target": [0, 0.6, 0.65],
                "name": "East Side View"
            },
            "camera_3_south": {
                "position": [0, 2.5, 1.0],
                "target": [0, 0.6, 0.65],
                "name": "South Side View (Arm 2 Perspective)"
            },
            "camera_4_west": {
                "position": [-1.5, 0.6, 1.0],
                "target": [0, 0.6, 0.65],
                "name": "West Side View"
            },
            "camera_5_overhead": {
                "position": [0, 0.6, 2.0],
                "target": [0, 0.6, 0.65],
                "name": "Overhead View (Game Board)"
            }
        }
    
    def _setup_environment(self):
        """Setup the physical environment: plane, table, and game board"""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table at center
        table_pos = [0, 1.0, 0]
        self.table_id = p.loadURDF("table/table.urdf", table_pos, 
                                   p.getQuaternionFromEuler([0, 0, 0]))
        
        # Create a visual tic-tac-toe board on the table
        board_pos = [0, 0.6, 0.66]  # Slightly above table surface
        self.board_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.15, 0.15, 0.005],
                rgbaColor=[1, 1, 1, 1]  # White board
            ),
            basePosition=board_pos
        )
        
        # Add grid lines (visual markers for tic-tac-toe)
        self._create_board_grid_markers()
        
        print("✅ Environment setup complete: plane, table, and game board loaded")
    
    def _create_board_grid_markers(self):
        """Create visual grid lines for the tic-tac-toe board"""
        board_center = [0, 0.6, 0.67]
        grid_size = 0.1
        line_thickness = 0.002
        
        # Vertical lines
        for i in [-1, 0, 1]:
            x_offset = i * grid_size / 3
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[line_thickness, grid_size, 0.001],
                    rgbaColor=[0, 0, 0, 1]  # Black lines
                ),
                basePosition=[board_center[0] + x_offset, board_center[1], board_center[2]]
            )
        
        # Horizontal lines  
        for i in [-1, 0, 1]:
            y_offset = i * grid_size / 3
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[grid_size, line_thickness, 0.001],
                    rgbaColor=[0, 0, 0, 1]  # Black lines
                ),
                basePosition=[board_center[0], board_center[1] + y_offset, board_center[2]]
            )
    
    def _setup_robots(self):
        """Setup two KUKA IIWA robotic arms on opposite sides"""
        # Robot Arm 1 (Player X) - North side of table
        arm1_pos = [0, -0.5, 0]
        arm1_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_arm1_id = p.loadURDF("kuka_iiwa/model.urdf", arm1_pos, 
                                        arm1_orientation, useFixedBase=True)
        
        # Robot Arm 2 (Player O) - South side of table  
        arm2_pos = [0, 1.7, 0]
        arm2_orientation = p.getQuaternionFromEuler([0, 0, math.pi])  # Rotated 180 degrees
        self.robot_arm2_id = p.loadURDF("kuka_iiwa/model.urdf", arm2_pos, 
                                        arm2_orientation, useFixedBase=True)
        
        # Get robot information
        self.num_joints_arm1 = p.getNumJoints(self.robot_arm1_id)
        self.num_joints_arm2 = p.getNumJoints(self.robot_arm2_id)
        self.ee_index = 6  # End-effector index for KUKA IIWA
        
        print(f"✅ Robots loaded: Arm1 ({self.num_joints_arm1} joints), Arm2 ({self.num_joints_arm2} joints)")
    
    def _initialize_camera_windows(self):
        """Initialize OpenCV windows for all five cameras"""
        for cam_name, cam_config in self.camera_configs.items():
            window_name = f"Camera: {cam_config['name']}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 480, 360)
            self.camera_windows[cam_name] = window_name
        
        print("✅ Camera windows initialized")
    
    def get_camera_image(self, camera_key):
        """Capture image from specified camera"""
        cam_config = self.camera_configs[camera_key]
        
        width, height = 640, 480
        fov, aspect = 60, width / height
        near, far = 0.01, 10
        
        # Calculate view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_config["position"],
            cameraTargetPosition=cam_config["target"],
            cameraUpVector=[0, 0, 1]
        )
        
        # Calculate projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=near, farVal=far
        )
        
        # Capture image
        images = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        # Convert to OpenCV format
        rgb_array = np.array(images[2])
        rgb_image = rgb_array.reshape(height, width, 4)[:, :, :3]
        
        return rgb_image, view_matrix, projection_matrix
    
    def update_camera_displays(self):
        """Update all camera displays with current images"""
        for cam_key, window_name in self.camera_windows.items():
            rgb_image, _, _ = self.get_camera_image(cam_key)
            
            # Convert RGB to BGR for OpenCV display
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Add camera label
            cv2.putText(bgr_image, self.camera_configs[cam_key]["name"], 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display image
            cv2.imshow(window_name, bgr_image)
    
    def move_arm_to_position(self, arm_id, target_position, duration=2.0):
        """Move specified arm to target position using inverse kinematics"""
        if not isinstance(target_position, (list, tuple)) or len(target_position) != 3:
            print(f"❌ Invalid target position: {target_position}")
            return False
        
        # Calculate inverse kinematics
        ee_orientation = p.getQuaternionFromEuler([0, -math.pi/2, 0])
        joint_positions = p.calculateInverseKinematics(
            arm_id, self.ee_index, target_position, ee_orientation
        )
        
        # Get current joint positions
        current_positions = [p.getJointState(arm_id, i)[0] 
                           for i in range(len(joint_positions))]
        
        # Smooth interpolation
        steps = int(duration * 240)  # 240 Hz simulation
        for step in range(steps):
            alpha = step / steps
            interpolated_positions = [
                current_positions[i] + alpha * (joint_positions[i] - current_positions[i])
                for i in range(len(joint_positions))
            ]
            
            # Apply joint positions
            for i in range(len(joint_positions)):
                p.setJointMotorControl2(
                    arm_id, i, p.POSITION_CONTROL,
                    targetPosition=interpolated_positions[i],
                    force=200
                )
            
            # Step simulation and update displays
            p.stepSimulation()
            self.update_camera_displays()
            
            # Check for ESC key to exit
            if cv2.waitKey(1) & 0xFF == 27:
                return False
            
            time.sleep(1/240)
        
        return True
    
    def demonstrate_basic_movements(self):
        """Demonstrate basic arm movements for testing"""
        print("🎯 Starting basic movement demonstration...")
        
        # Define some test positions around the board
        board_center = [0, 0.6, 0.8]
        positions = {
            "arm1_home": [0.3, 0.2, 0.8],
            "arm1_point_center": [0, 0.6, 0.75],
            "arm1_point_corner": [-0.1, 0.5, 0.75],
            "arm2_home": [-0.3, 1.0, 0.8], 
            "arm2_point_center": [0, 0.6, 0.75],
            "arm2_point_corner": [0.1, 0.7, 0.75]
        }
        
        # Test Arm 1 movements
        print("🔹 Testing Arm 1 (Player X) movements...")
        self.move_arm_to_position(self.robot_arm1_id, positions["arm1_home"])
        time.sleep(1)
        self.move_arm_to_position(self.robot_arm1_id, positions["arm1_point_center"])
        time.sleep(1)
        self.move_arm_to_position(self.robot_arm1_id, positions["arm1_point_corner"])
        time.sleep(1)
        self.move_arm_to_position(self.robot_arm1_id, positions["arm1_home"])
        
        # Test Arm 2 movements
        print("🔹 Testing Arm 2 (Player O) movements...")
        self.move_arm_to_position(self.robot_arm2_id, positions["arm2_home"])
        time.sleep(1)
        self.move_arm_to_position(self.robot_arm2_id, positions["arm2_point_center"])
        time.sleep(1)
        self.move_arm_to_position(self.robot_arm2_id, positions["arm2_point_corner"])
        time.sleep(1)
        self.move_arm_to_position(self.robot_arm2_id, positions["arm2_home"])
        
        print("✅ Basic movement demonstration complete!")
    
    def run_demo(self):
        """Main demo loop"""
        print("🚀 Starting Multi-Robot Tic-Tac-Toe Environment Demo")
        print("📹 Five camera views should now be visible")
        print("🎮 Controls: ESC to exit, SPACE to run movement demo")
        
        try:
            while True:
                # Update simulation
                p.stepSimulation()
                
                # Update all camera displays
                self.update_camera_displays()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("🛑 ESC pressed - Exiting...")
                    break
                elif key == ord(' '):  # SPACE key
                    print("🎯 SPACE pressed - Running movement demonstration...")
                    self.demonstrate_basic_movements()
                
                # Small delay for smooth operation
                time.sleep(1/60)
                
        except KeyboardInterrupt:
            print("🛑 Keyboard interrupt received")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        cv2.destroyAllWindows()
        p.disconnect()
        print("✅ Cleanup complete!")

def main():
    """Main function"""
    print("=" * 60)
    print("🤖 MULTI-ROBOT TIC-TAC-TOE: DUAL ARM ENVIRONMENT")
    print("📋 Program 1: Foundation Setup with Five Virtual Cameras")
    print("=" * 60)
    
    # Create and run the environment
    env = DualArmEnvironment()
    env.run_demo()

if __name__ == "__main__":
    main()
