#!/usr/bin/env python3
"""
Multi-Robot Tic-Tac-Toe: Dual Arm Environment with Five Virtual Cameras
Program 1 (Revised): Fixed arm positioning, centered board, controlled camera updates
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
        
        # Game board specifications (DEFINE FIRST)
        self.table_center = [0, 0.6, 0.625]  # Properly centered on table surface
        self.board_size = 0.3  # 30cm x 30cm board
        
        # Camera control
        self.camera_configs = self._setup_camera_configs()
        self.camera_windows = {}
        self.camera_update_counter = 0
        self.camera_update_interval = 10  # Update every 10 frames (slower)
        self.current_camera_focus = "camera_5_overhead"  # Start with overhead view
        
        # Initialize the environment
        self._setup_environment()
        self._setup_robots()
        self._initialize_camera_windows()
        
        # Quick optimal view setup
        p.resetDebugVisualizerCamera(2.2, 35, -30, [0, 0.6, 0.65])
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # Set optimal viewing angle for comprehensive view
        self.setup_optimal_camera_view()
        self.setup_multiple_viewpoints()
        self.current_viewpoint_index = 0
        
        print("✅ Dual-arm environment initialized successfully!")
        print("📋 ROS2 + Gazebo ecosystem noted for future integration!")

    def cycle_viewpoint(self):
        """Cycle through different optimal viewpoints"""
        viewpoint_names = list(self.viewpoints.keys())
        current_name = viewpoint_names[self.current_viewpoint_index]
        params = self.viewpoints[current_name]
    
        # Apply viewpoint
        p.resetDebugVisualizerCamera(
            cameraDistance=params["distance"],
            cameraYaw=params["yaw"],
            cameraPitch=params["pitch"], 
            cameraTargetPosition=self.table_center
        )
    
        print(f"📹 Switched to {current_name.upper()} view")
    
        # Move to next viewpoint for next press
        self.current_viewpoint_index = (self.current_viewpoint_index + 1) % len(viewpoint_names)
    
    def _setup_camera_configs(self):
        """Configure five virtual cameras with optimal viewing angles"""
        board_center = self.table_center
        
        return {
            "camera_1_north": {
                "position": [0, board_center[1] - 0.8, 1.2],
                "target": board_center,
                "name": "North Side View (Arm 1 Perspective)"
            },
            "camera_2_east": {
                "position": [0.8, board_center[1], 1.2],
                "target": board_center,
                "name": "East Side View"
            },
            "camera_3_south": {
                "position": [0, board_center[1] + 0.8, 1.2],
                "target": board_center,
                "name": "South Side View (Arm 2 Perspective)"
            },
            "camera_4_west": {
                "position": [-0.8, board_center[1], 1.2],
                "target": board_center,
                "name": "West Side View"
            },
            "camera_5_overhead": {
                "position": [0, board_center[1], 1.8],
                "target": board_center,
                "name": "Overhead View (Game Board)"
            }
        }
    
    def setup_optimal_camera_view(self):
        """Set up optimal viewing angle for comprehensive scene overview"""
        # Calculate optimal camera position based on scene bounds
        table_center = self.table_center
    
        # Position camera for best comprehensive view
        optimal_distance = 2.5  # Distance from target
        optimal_yaw = 45        # 45-degree angle for good arm visibility
        optimal_pitch = -25     # Slightly above for overview
    
        # Set the camera to focus on the table center with good arm visibility
        p.resetDebugVisualizerCamera(
            cameraDistance=optimal_distance,
            cameraYaw=optimal_yaw, 
            cameraPitch=optimal_pitch,
            cameraTargetPosition=[table_center[0], table_center[1], table_center[2]]
        )
    
        # Optional: Disable some GUI elements for cleaner view
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Keep GUI
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)  # Better rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # Enable rendering
    
        print(f"📹 Optimal camera view set: Distance={optimal_distance}, Yaw={optimal_yaw}°, Pitch={optimal_pitch}°")

    def setup_multiple_viewpoints(self):
        """Setup hotkeys for different optimal viewpoints"""
        viewpoints = {
            "overview": {"distance": 2.5, "yaw": 45, "pitch": -25},     # General overview
            "close": {"distance": 1.5, "yaw": 30, "pitch": -20},       # Close-up action
            "side": {"distance": 2.0, "yaw": 90, "pitch": -15},        # Side view of arms
            "top": {"distance": 2.2, "yaw": 0, "pitch": -60},          # Top-down view
            "dramatic": {"distance": 3.0, "yaw": 60, "pitch": -35}     # Dramatic angle
        }
    
        self.viewpoints = viewpoints
        print("📹 Multiple viewpoints configured:")
        print("   Press V during simulation to cycle through viewpoints")
        for name, params in viewpoints.items():
            print(f"   {name.capitalize()}: Distance={params['distance']}, Yaw={params['yaw']}°, Pitch={params['pitch']}°")
    
    def _setup_environment(self):
        """Setup the physical environment: plane, table, and properly centered game board"""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table at center
        table_pos = [0, 0.6, 0]  # Table position
        self.table_id = p.loadURDF("table/table.urdf", table_pos, 
                                   p.getQuaternionFromEuler([0, 0, 0]))
        
        # Create a properly centered tic-tac-toe board on the table
        board_pos = [self.table_center[0], self.table_center[1], self.table_center[2] + 0.01]
        self.board_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[self.board_size/2, self.board_size/2, 0.005],
                rgbaColor=[0.9, 0.9, 0.9, 1]  # Light gray board
            ),
            basePosition=board_pos
        )
        
        # Add centered grid lines
        self._create_board_grid_markers()
        
        # Add visual markers for board corners (for debugging)
        self._add_corner_markers()
        
        print(f"✅ Environment setup complete: board centered at {self.table_center}")
    
    def _create_board_grid_markers(self):
        """Create properly centered visual grid lines for the tic-tac-toe board"""
        board_center = self.table_center
        grid_spacing = self.board_size / 3  # Divide board into 3x3 grid
        line_thickness = 0.003
        line_height = 0.002
        
        # Vertical lines (2 lines to create 3 columns)
        for i in [-1, 1]:  # Lines at -1/3 and +1/3 positions
            x_offset = i * grid_spacing / 3
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[line_thickness, self.board_size/2, line_height],
                    rgbaColor=[0, 0, 0, 1]  # Black lines
                ),
                basePosition=[board_center[0] + x_offset, board_center[1], 
                             board_center[2] + line_height]
            )
        
        # Horizontal lines (2 lines to create 3 rows)
        for i in [-1, 1]:  # Lines at -1/3 and +1/3 positions
            y_offset = i * grid_spacing / 3
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[self.board_size/2, line_thickness, line_height],
                    rgbaColor=[0, 0, 0, 1]  # Black lines
                ),
                basePosition=[board_center[0], board_center[1] + y_offset, 
                             board_center[2] + line_height]
            )
    
    def _add_corner_markers(self):
        """Add small colored markers at board corners for positioning verification"""
        corners = [
            [-self.board_size/2, -self.board_size/2],  # Bottom-left
            [self.board_size/2, -self.board_size/2],   # Bottom-right
            [self.board_size/2, self.board_size/2],    # Top-right
            [-self.board_size/2, self.board_size/2]    # Top-left
        ]
        
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]  # RGBY
        
        for i, (corner, color) in enumerate(zip(corners, colors)):
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.01,
                    rgbaColor=color
                ),
                basePosition=[self.table_center[0] + corner[0], 
                             self.table_center[1] + corner[1],
                             self.table_center[2] + 0.02]
            )
    
    def _setup_robots(self):
        """Setup two compact arms positioned for dramatic human-like movements"""
        # Robot Arm 1 (Player X) - Positioned farther back for dramatic reach
        arm1_pos = [-0.3, self.table_center[1] - 0.45, 0.625]  # Back and to the side
        arm1_orientation = p.getQuaternionFromEuler([0, 0, math.pi/6])  # 30-degree angle
    
        self.robot_arm1_id = p.loadURDF("kuka_iiwa/model.urdf", arm1_pos, 
                                        arm1_orientation, useFixedBase=True,
                                        globalScaling=0.5)  # Medium scale for reach
        self.ee_index = 6
    
        # Robot Arm 2 (Player O) - Positioned farther back on opposite side
        arm2_pos = [0.3, self.table_center[1] + 0.45, 0.625]   # Back and to the side
        arm2_orientation = p.getQuaternionFromEuler([0, 0, math.pi + math.pi/6])  # 210-degree angle
    
        self.robot_arm2_id = p.loadURDF("kuka_iiwa/model.urdf", arm2_pos, 
                                        arm2_orientation, useFixedBase=True,
                                        globalScaling=0.5)  # Medium scale for reach
    
        # Get robot information
        self.num_joints_arm1 = p.getNumJoints(self.robot_arm1_id)
        self.num_joints_arm2 = p.getNumJoints(self.robot_arm2_id)
    
        # Add pointing hand visuals
        self._add_pointing_hands()
    
        # Move arms to dramatic ready positions
        self._move_arms_to_ready_position()
    
        print(f"✅ Dramatic positioning arms: Arm1 at {arm1_pos}, Arm2 at {arm2_pos}")
        print(f"   Arms positioned for challenging, human-like movements")
    
    def _move_arms_to_ready_position(self):
        """Move arms to varied ready positions that exercise all joints"""
        # Arm 1: Dramatic ready position using all joints
        ready_pos_arm1 = [-0.25, self.table_center[1] - 0.35, 0.90]
        ready_orient_arm1 = p.getQuaternionFromEuler([math.pi/6, -math.pi/4, math.pi/8])
    
        # Arm 2: Different dramatic ready position
        ready_pos_arm2 = [0.25, self.table_center[1] + 0.35, 0.90]  
        ready_orient_arm2 = p.getQuaternionFromEuler([-math.pi/6, math.pi/4, -math.pi/8])
    
        # Set positions with full IK constraints
        for arm_id, pos, orient, arm_name in [(self.robot_arm1_id, ready_pos_arm1, ready_orient_arm1, "Arm1"), 
                                              (self.robot_arm2_id, ready_pos_arm2, ready_orient_arm2, "Arm2")]:
        
            # Calculate IK with constraints to use all joints
            joint_positions = p.calculateInverseKinematics(
                arm_id, self.ee_index, pos, orient,
                lowerLimits=[-2.9, -2.1, -2.9, -2.1, -2.9, -2.1, -3.0],
                upperLimits=[2.9, 2.1, 2.9, 2.1, 2.9, 2.1, 3.0],
                jointRanges=[5.8, 4.2, 5.8, 4.2, 5.8, 4.2, 6.0],
                restPoses=[0.5, -0.8, 0.3, -1.2, -0.5, 1.3, 0.2]  # Varied rest poses
            )
        
            # Apply to all 7 joints with different parameters
            for i in range(min(len(joint_positions), 7)):
                p.setJointMotorControl2(
                    arm_id, i, p.POSITION_CONTROL,
                    targetPosition=joint_positions[i], 
                    force=400 + i * 50,  # Varied force per joint
                    maxVelocity=8,
                    positionGain=0.3
                )
        
            print(f"✅ {arm_name} ready position set - using all 7 joints")
    
        # Extended settling time to see all joints move
        for step in range(200):
            p.stepSimulation()
            if step % 50 == 0:  # Monitor every 50 steps
                print(f"🔧 Settling step {step}/200")
            time.sleep(1/240)
    
    def test_all_joints_movement(self):
        """Test function to verify all 7 joints can move independently"""
        print("🧪 Testing all 7 joints independently...")
    
        for arm_id, arm_name in [(self.env.robot_arm1_id, "Arm1"), (self.env.robot_arm2_id, "Arm2")]:
            print(f"\n🔧 Testing {arm_name}:")
        
            # Get current positions
            initial_positions = [p.getJointState(arm_id, i)[0] for i in range(7)]
        
            # Test each joint individually
            for joint_idx in range(7):
                print(f"  Testing Joint {joint_idx + 1}...")
            
                # Move only this joint
                test_positions = initial_positions.copy()
                test_positions[joint_idx] += 0.3  # Move 0.3 radians
            
                # Apply movement
                for i in range(7):
                    target_pos = test_positions[i] if i == joint_idx else initial_positions[i]
                    p.setJointMotorControl2(
                        arm_id, i, p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=500,
                        maxVelocity=5
                    )
            
                # Let it move
                for _ in range(60):
                    p.stepSimulation()
                    time.sleep(1/240)
            
                # Check if it moved
                final_pos = p.getJointState(arm_id, joint_idx)[0]
                moved = abs(final_pos - initial_positions[joint_idx]) > 0.1
                status = "✅ MOVED" if moved else "❌ STUCK"
                print(f"    Joint {joint_idx + 1}: {status} ({initial_positions[joint_idx]:.3f} → {final_pos:.3f})")
        
            # Return to initial positions
            for i in range(7):
                p.setJointMotorControl2(
                    arm_id, i, p.POSITION_CONTROL,
                    targetPosition=initial_positions[i],
                    force=400
                )
        
            for _ in range(100):
                p.stepSimulation()
    
    def _initialize_camera_windows(self):
        """Initialize OpenCV windows with better layout"""
        # Create windows with specific positions
        positions = [(100, 100), (600, 100), (1100, 100), (100, 500), (600, 500)]
        
        for i, (cam_name, cam_config) in enumerate(self.camera_configs.items()):
            window_name = f"Camera: {cam_config['name']}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 400, 300)
            if i < len(positions):
                cv2.moveWindow(window_name, positions[i][0], positions[i][1])
            self.camera_windows[cam_name] = window_name
        
        print("✅ Camera windows arranged in optimal layout")
    
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
        """Update camera displays with controlled refresh rate"""
        self.camera_update_counter += 1
        
        # Only update cameras every N frames to reduce flickering
        if self.camera_update_counter % self.camera_update_interval != 0:
            return
            
        for cam_key, window_name in self.camera_windows.items():
            try:
                rgb_image, _, _ = self.get_camera_image(cam_key)
                
                # Convert RGB to BGR for OpenCV display
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                # Add informative overlay
                camera_name = self.camera_configs[cam_key]["name"]
                cv2.putText(bgr_image, camera_name, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Highlight current focus camera
                if cam_key == self.current_camera_focus:
                    cv2.rectangle(bgr_image, (0, 0), (639, 479), (0, 255, 255), 3)
                    cv2.putText(bgr_image, "FOCUS", (10, 460), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add grid position indicators for overhead camera
                if cam_key == "camera_5_overhead":
                    self._add_grid_overlay(bgr_image)
                
                # Display image
                cv2.imshow(window_name, bgr_image)
                
            except Exception as e:
                print(f"⚠️ Camera {cam_key} error: {e}")
    
    def _add_grid_overlay(self, image):
        """Add tic-tac-toe grid position overlays to overhead camera"""
        height, width = image.shape[:2]
        
        # Draw grid lines on the image for reference
        for i in range(1, 3):
            # Vertical lines
            x = int(width * i / 3)
            cv2.line(image, (x, 0), (x, height), (255, 255, 0), 1)
            
            # Horizontal lines
            y = int(height * i / 3)
            cv2.line(image, (0, y), (width, y), (255, 255, 0), 1)
        
        # Add position labels
        positions = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        for i, pos in enumerate(positions):
            x = int(width * (i % 3 + 0.5) / 3)
            y = int(height * (i // 3 + 0.5) / 3)
            cv2.putText(image, pos, (x-10, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def get_board_positions(self):
        """Get 3D coordinates for finger pointing to tic-tac-toe positions"""
        positions = {}
        spacing = self.board_size / 3
    
        for row in range(3):
            for col in range(3):
                pos_num = row * 3 + col + 1
                x = self.table_center[0] + (col - 1) * spacing
                y = self.table_center[1] + (row - 1) * spacing
                z = self.table_center[2] + 0.05  # Just above board for finger pointing
                positions[pos_num] = [x, y, z]
    
        return positions
    
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
    
    def _add_pointing_hands(self):
        """Add visual pointing hand indicators at end-effector positions"""
        # Create simple hand/finger pointing indicators
        hand_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.015,  # 1.5cm radius finger
            length=0.08,   # 8cm long finger
            rgbaColor=[0.9, 0.7, 0.6, 1.0]  # Skin-like color
        )
    
        # Store the visual shape for later use
        self.hand_visual_shape = hand_visual_shape
        self.pointing_hand_ids = {}  # Store hand visualization IDs
    
        print("✅ Pointing hand visualizations prepared")

    def demonstrate_board_coverage(self):
        """Demonstrate that both arms can reach all board positions"""
        print("🎯 Demonstrating board coverage...")
        board_positions = self.get_board_positions()
        
        # Test Arm 1 reaching corners and center
        print("🔹 Arm 1 (North) reaching board positions...")
        test_positions_arm1 = [1, 2, 3, 5]  # Top row + center
        for pos in test_positions_arm1:
            print(f"   Arm 1 → Position {pos}")
            if not self.move_arm_to_position(self.robot_arm1_id, board_positions[pos], 1.5):
                return False
            time.sleep(0.5)
        
        # Return arm 1 to ready
        ready_pos_arm1 = [0.3, self.table_center[1] - 0.3, 0.9]
        self.move_arm_to_position(self.robot_arm1_id, ready_pos_arm1, 1.0)
        
        # Test Arm 2 reaching corners and center
        print("🔹 Arm 2 (South) reaching board positions...")
        test_positions_arm2 = [7, 8, 9, 5]  # Bottom row + center
        for pos in test_positions_arm2:
            print(f"   Arm 2 → Position {pos}")
            if not self.move_arm_to_position(self.robot_arm2_id, board_positions[pos], 1.5):
                return False
            time.sleep(0.5)
        
        # Return arm 2 to ready
        ready_pos_arm2 = [0.3, self.table_center[1] + 0.3, 0.9]
        self.move_arm_to_position(self.robot_arm2_id, ready_pos_arm2, 1.0)
        
        print("✅ Board coverage demonstration complete!")
        return True
    
    def run_demo(self):
        """Main demo loop with improved controls"""
        print("🚀 Starting Multi-Robot Tic-Tac-Toe Environment Demo")
        print("📹 Five camera views are now visible with controlled refresh")
        print("🎮 Controls:")
        print("   ESC - Exit program")
        print("   SPACE - Run board coverage demonstration")
        print("   1-5 - Focus on specific camera (1=North, 2=East, 3=South, 4=West, 5=Overhead)")
        print("   + - Increase camera refresh rate")
        print("   - - Decrease camera refresh rate")
        
        try:
            while True:
                # Update simulation
                p.stepSimulation()
                
                # Update camera displays with controlled rate
                self.update_camera_displays()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("🛑 ESC pressed - Exiting...")
                    break
                elif key == ord(' '):  # SPACE key
                    print("🎯 Running board coverage demonstration...")
                    self.demonstrate_board_coverage()
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    camera_keys = list(self.camera_configs.keys())
                    idx = int(chr(key)) - 1
                    if idx < len(camera_keys):
                        self.current_camera_focus = camera_keys[idx]
                        print(f"📹 Focusing on: {self.camera_configs[camera_keys[idx]]['name']}")
                elif key == ord('+'):
                    self.camera_update_interval = max(1, self.camera_update_interval - 2)
                    print(f"📹 Camera refresh rate increased (interval: {self.camera_update_interval})")
                elif key == ord('-'):
                    self.camera_update_interval = min(30, self.camera_update_interval + 2)
                    print(f"📹 Camera refresh rate decreased (interval: {self.camera_update_interval})")
                
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
    print("=" * 70)
    print("🤖 MULTI-ROBOT TIC-TAC-TOE: DUAL ARM ENVIRONMENT (REVISED)")
    print("📋 Program 1: Fixed positioning, centered board, controlled cameras")
    print("=" * 70)
    
    # Create and run the environment
    env = DualArmEnvironment()
    env.run_demo()

if __name__ == "__main__":
    main()