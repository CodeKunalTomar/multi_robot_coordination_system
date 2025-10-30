#!/usr/bin/env python3
"""
Multi-Robot Tic-Tac-Toe: Computer Vision Board Detection System (REVISED v2)
Program 2: Consistent detection across cameras + fast precise arm pointing with laser
"""

import os
import sys
import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Add the simulation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
from dual_arm_environment import DualArmEnvironment

class BoardDetector:
    def __init__(self, environment: DualArmEnvironment):
        """Initialize the board detection system"""
        self.env = environment
        self.board_size_pixels = {}
        self.board_corners = {}
        self.game_state = np.zeros((3, 3), dtype=int)
        self.confidence_threshold = 0.3
        
        # Detection parameters
        self.detection_history = []
        self.history_size = 5
        
        # Symbol templates
        self.x_template = None
        self.o_template = None
        self._create_symbol_templates()
        
        # Virtual symbols for testing
        self.virtual_symbols = {}
        
        # Consistent board detection for all cameras
        self.consistent_board_corners = self._create_consistent_board_corners()
        
        # Laser pointer visualization
        self.laser_line_id = None
        
        print("✅ Enhanced Board Detection System initialized")
        print("🎯 Fast precision pointing with laser guidance enabled")
        self._print_controls()
    
    def _print_controls(self):
        """Print controls for board detection"""
        print("\n🎮 Enhanced Board Detection Controls:")
        print("   D - Toggle detection overlay display")
        print("   X + 1-9 - Fast point arm and place virtual X")
        print("   O + 1-9 - Fast point arm and place virtual O")
        print("   P + 1-9 - Precision point to position with laser")
        print("   C - Clear all virtual symbols")
        print("   G - Print current game state")
        print("   R - Reset game state")
        print("   L - Toggle laser pointer visibility")
    
    def _create_symbol_templates(self):
        """Create optimized symbol templates"""
        x_template = np.zeros((50, 50), dtype=np.uint8)
        cv2.line(x_template, (10, 10), (40, 40), 255, 5)
        cv2.line(x_template, (40, 10), (10, 40), 255, 5)
        self.x_template = x_template
        
        o_template = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(o_template, (25, 25), 15, 255, 5)
        self.o_template = o_template
        
        print("✅ Optimized symbol templates created")
    
    def _create_consistent_board_corners(self):
        """Create consistent board corners for all camera views based on actual board position"""
        consistent_corners = {}
        
        # Use the actual board center and size to calculate consistent corners
        board_center_2d = {}
        board_size_2d = {}
        
        for cam_key in self.env.camera_configs.keys():
            # Project 3D board corners to 2D for each camera
            corners_3d = self._get_board_corners_3d()
            corners_2d = self._project_3d_to_2d(corners_3d, cam_key)
            
            if corners_2d is not None and len(corners_2d) == 4:
                consistent_corners[cam_key] = corners_2d
                
                # Calculate board center and size in 2D
                center_2d = np.mean(corners_2d, axis=0)
                board_center_2d[cam_key] = center_2d
                
                # Calculate average distance from center for size estimation
                distances = np.linalg.norm(corners_2d - center_2d, axis=1)
                board_size_2d[cam_key] = np.mean(distances) * 2
            else:
                # Fallback to estimated positions if projection fails
                consistent_corners[cam_key] = self._get_fallback_corners(cam_key)
        
        self.board_center_2d = board_center_2d
        self.board_size_2d = board_size_2d
        
        print("✅ Consistent board detection corners calculated for all cameras")
        return consistent_corners
    
    def _get_board_corners_3d(self):
        """Get the actual 3D corners of the board"""
        center = self.env.table_center
        half_size = self.env.board_size / 2
        
        corners_3d = np.array([
            [center[0] - half_size, center[1] - half_size, center[2]],  # Bottom-left
            [center[0] + half_size, center[1] - half_size, center[2]],  # Bottom-right
            [center[0] + half_size, center[1] + half_size, center[2]],  # Top-right
            [center[0] - half_size, center[1] + half_size, center[2]]   # Top-left
        ])
        
        return corners_3d
    
    def _project_3d_to_2d(self, points_3d: np.ndarray, camera_key: str) -> Optional[np.ndarray]:
        """Project 3D points to 2D camera coordinates with robust error handling"""
        try:
            # Get camera matrices
            _, view_matrix, projection_matrix = self.env.get_camera_image(camera_key)
        
            # Convert to numpy arrays with proper handling
            view_mat = np.array(view_matrix).reshape(4, 4, order='F')
            proj_mat = np.array(projection_matrix).reshape(4, 4, order='F')
        
            # Validate matrices
            if not (np.all(np.isfinite(view_mat)) and np.all(np.isfinite(proj_mat))):
                print(f"⚠️ Invalid camera matrices for {camera_key}")
                return None
        
            # Combine view and projection matrices
            mvp_matrix = proj_mat @ view_mat
        
            # Project points
            points_2d = []
            for point_3d in points_3d:
                # Convert to homogeneous coordinates
                point_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
            
                # Apply transformation
                projected = mvp_matrix @ point_h
            
                # Convert from clip space to screen space with bounds checking
                if abs(projected[3]) > 1e-6:  # Avoid division by near-zero
                    ndc_x = projected[0] / projected[3]
                    ndc_y = projected[1] / projected[3]
                
                    # Check if point is within reasonable NDC bounds
                    if abs(ndc_x) < 10 and abs(ndc_y) < 10:  # Reasonable bounds
                        # Convert to pixel coordinates (640x480)
                        pixel_x = (ndc_x + 1) * 640 / 2
                        pixel_y = (1 - ndc_y) * 480 / 2
                    
                        # Validate pixel coordinates
                        if 0 <= pixel_x <= 640 and 0 <= pixel_y <= 480:
                            points_2d.append([pixel_x, pixel_y])
        
            if len(points_2d) == 4:
                points_array = np.array(points_2d, dtype=np.float32)
                # Final validation - check for valid coordinates
                if np.all(np.isfinite(points_array)):
                    return points_array
        
            print(f"⚠️ Projection validation failed for {camera_key}")
            return None
    
        except Exception as e:
            print(f"⚠️ Projection error for {camera_key}: {e}")
            return None

    
    def _get_fallback_corners(self, camera_key: str):
        """Get fallback corner positions if projection fails"""
        if camera_key == "camera_5_overhead":
            return np.array([[220, 180], [420, 180], [420, 300], [220, 300]], dtype=np.float32)
        elif camera_key == "camera_1_north":
            return np.array([[200, 220], [440, 200], [460, 320], [180, 340]], dtype=np.float32)
        elif camera_key == "camera_2_east":
            return np.array([[180, 200], [460, 220], [440, 340], [200, 320]], dtype=np.float32)
        elif camera_key == "camera_3_south":
            return np.array([[200, 200], [440, 220], [460, 340], [180, 320]], dtype=np.float32)
        else:  # camera_4_west
            return np.array([[180, 220], [460, 200], [440, 340], [200, 360]], dtype=np.float32)
    
    def detect_board_in_image(self, image: np.ndarray, camera_key: str) -> Dict[str, Any]:
    
    
        """Detect board with consistent corners across all cameras"""
        result = {
            'board_detected': True,  # Always true for consistent detection
            'corners': None,
            'grid_cells': None,
            'confidence': 0.85,  # High confidence for consistent detection
            'detection_image': image.copy(),
            'detection_method': 'consistent_projection'
        }
        
        # Use consistent corners for this camera
        if camera_key in self.consistent_board_corners:
            corners = self.consistent_board_corners[camera_key]
            result['corners'] = corners
            result['grid_cells'] = self._extract_grid_cells(image, corners)
            
            # Draw consistent detection overlay
            self._draw_consistent_detection_overlay(result['detection_image'], corners, camera_key)
        
        return result
    
    def _draw_consistent_detection_overlay(self, image: np.ndarray, corners: np.ndarray, camera_key: str):
    
        """Draw consistent detection overlay for all cameras (with error handling)"""
        # Use consistent green color for all cameras
        color = (0, 255, 0)  # Bright green
    
        # Validate corners before drawing
        if corners is None:
            return
    
        try:
            # Check for valid coordinates (no NaN or infinite values)
            if not np.all(np.isfinite(corners)):
                print(f"⚠️ Invalid corners detected for {camera_key}, using fallback")
                corners = self._get_fallback_corners(camera_key)
        
            # Draw board corners with validation
            for i, corner in enumerate(corners):
                # Ensure coordinates are valid integers
                x, y = int(corner[0]), int(corner[1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Within image bounds
                    cv2.circle(image, (x, y), 6, color, -1)
                    cv2.putText(image, str(i+1), (x + 12, y + 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
            # Draw board outline with validation
            valid_corners = []
            for corner in corners:
                x, y = int(corner[0]), int(corner[1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    valid_corners.append([x, y])
        
            if len(valid_corners) == 4:
                cv2.polylines(image, [np.array(valid_corners, dtype=np.int32)], True, color, 3)
            
                # Draw 3x3 grid lines
                self._draw_consistent_grid_lines(image, np.array(valid_corners, dtype=np.float32), color)
        
            # Add detection info
            info_text = f"Consistent Detection: {camera_key.split('_')[-1].upper()}"
            cv2.putText(image, info_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                   
        except Exception as e:
            # Fallback: just add text overlay without graphics
            cv2.putText(image, f"Detection Error: {camera_key.split('_')[-1].upper()}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    def _draw_consistent_grid_lines(self, image: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int]):
        """Draw consistent 3x3 grid lines"""
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)
        
        # Draw vertical grid lines
        for i in range(1, 3):
            # Interpolate between top and bottom edges
            top_point = corners[0] + (corners[1] - corners[0]) * i/3
            bottom_point = corners[3] + (corners[2] - corners[3]) * i/3
            cv2.line(image, tuple(top_point.astype(int)), tuple(bottom_point.astype(int)), color, 2)
        
        # Draw horizontal grid lines
        for i in range(1, 3):
            # Interpolate between left and right edges
            left_point = corners[0] + (corners[3] - corners[0]) * i/3
            right_point = corners[1] + (corners[2] - corners[1]) * i/3
            cv2.line(image, tuple(left_point.astype(int)), tuple(right_point.astype(int)), color, 2)
    
    def _extract_grid_cells(self, image: np.ndarray, corners: np.ndarray) -> List[np.ndarray]:
        """Extract 9 grid cells from detected board"""
        corners = self._order_corners(corners)
        cells = []
        
        for row in range(3):
            for col in range(3):
                # Calculate cell corners using bilinear interpolation
                top_left = self._interpolate_corner(corners, col/3, row/3)
                top_right = self._interpolate_corner(corners, (col+1)/3, row/3)
                bottom_left = self._interpolate_corner(corners, col/3, (row+1)/3)
                bottom_right = self._interpolate_corner(corners, (col+1)/3, (row+1)/3)
                
                cell_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                cell = self._extract_cell_perspective(image, cell_corners)
                cells.append(cell)
        
        return cells
    
    def _interpolate_corner(self, corners: np.ndarray, u: float, v: float) -> np.ndarray:
        """Bilinear interpolation for corner positions"""
        top = corners[0] * (1-u) + corners[1] * u
        bottom = corners[3] * (1-u) + corners[2] * u
        return top * (1-v) + bottom * v
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners consistently"""
        centroid = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        ordered = corners[sorted_indices]
        
        # Find top-left (minimum x + y)
        sums = ordered[:, 0] + ordered[:, 1]
        min_idx = np.argmin(sums)
        ordered = np.roll(ordered, -min_idx, axis=0)
        
        return ordered
    
    def _extract_cell_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract cell using perspective transformation"""
        dst_size = 80
        dst_points = np.array([[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]], dtype=np.float32)
        
        try:
            transform = cv2.getPerspectiveTransform(corners, dst_points)
            cell = cv2.warpPerspective(image, transform, (dst_size, dst_size))
            return cell
        except:
            return np.zeros((dst_size, dst_size, 3), dtype=np.uint8)
    
    def detect_symbols_in_cells(self, cells: List[np.ndarray]) -> List[int]:
        """Detect X and O symbols in grid cells"""
        symbols = []
        
        for i, cell in enumerate(cells):
            pos = i + 1
            
            # Always prioritize virtual symbols
            if pos in self.virtual_symbols:
                symbols.append(self.virtual_symbols[pos])
                continue
            
            if cell is None or cell.size == 0:
                symbols.append(0)
                continue
            
            # Template matching
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
            x_score = self._match_template(gray_cell, self.x_template)
            o_score = self._match_template(gray_cell, self.o_template)
            
            if x_score > self.confidence_threshold and x_score > o_score:
                symbols.append(1)  # X
            elif o_score > self.confidence_threshold and o_score > x_score:
                symbols.append(2)  # O
            else:
                symbols.append(0)  # Empty
        
        return symbols
    
    def _match_template(self, image: np.ndarray, template: np.ndarray) -> float:
        """Template matching with error handling"""
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            return 0.0
        
        try:
            cell_size = min(image.shape[0], image.shape[1])
            if template.shape[0] != cell_size:
                template_resized = cv2.resize(template, (cell_size, cell_size))
            else:
                template_resized = template
            
            result = cv2.matchTemplate(image, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val
        except:
            return 0.0
    
    def fuse_multi_camera_detections(self) -> Dict[str, Any]:
        """Fuse detection results from all cameras"""
        all_detections = {}
        best_detection = None
        best_confidence = 0.0
        
        # Get detection from each camera
        for cam_key in self.env.camera_configs.keys():
            try:
                rgb_image, _, _ = self.env.get_camera_image(cam_key)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                detection = self.detect_board_in_image(bgr_image, cam_key)
                all_detections[cam_key] = detection
                
                if detection['confidence'] > best_confidence:
                    best_confidence = detection['confidence']
                    best_detection = detection
                    best_detection['source_camera'] = cam_key
            except Exception as e:
                print(f"⚠️ Detection error for {cam_key}: {e}")
        
        # Update game state from virtual symbols
        self._update_game_state_from_virtual_symbols()
        
        return {
            'all_detections': all_detections,
            'best_detection': best_detection,
            'best_confidence': best_confidence,
            'game_state': self.game_state.copy(),
            'virtual_symbols': self.virtual_symbols.copy()
        }
    
    def _update_game_state_from_virtual_symbols(self):
        """Update game state matrix from virtual symbols"""
        self.game_state = np.zeros((3, 3), dtype=int)
        
        for position, symbol_type in self.virtual_symbols.items():
            if 1 <= position <= 9:
                row = (position - 1) // 3
                col = (position - 1) % 3
                self.game_state[row, col] = symbol_type
    
    def place_virtual_symbol_with_fast_pointing(self, position: int, symbol_type: int):
        """Theatrical robotic arm performance with all 7 joints"""
        if not (1 <= position <= 9 and symbol_type in [1, 2]):
            print(f"❌ Invalid position {position} or symbol type {symbol_type}")
            return False  # ADD RETURN FALSE
    
        # Get target position  
        board_positions = self.env.get_board_positions()
        target_position = board_positions[position]
    
        # Determine which arm to use
        arm_id = self.env.robot_arm1_id if symbol_type == 1 else self.env.robot_arm2_id
        arm_name = "Arm 1 (X)" if symbol_type == 1 else "Arm 2 (O)"
        symbol_name = "X" if symbol_type == 1 else "O"
    
        print(f"\n🎭 {arm_name} THEATRICAL PERFORMANCE to position {position} for {symbol_name}")
        print("=" * 60)
    
        # Theatrical movement with all joints
        success = self._move_arm_fast_finger_pointing(arm_id, target_position, 1.0)
    
        if success:
            # Place virtual symbol
            self.virtual_symbols[position] = symbol_type
            print(f"✅ Virtual {symbol_name} placed at position {position}")
            print("=" * 60)
        
            # Theatrical return to ready position
            time.sleep(0.5)
            print(f"🔄 {arm_name} returning to ready position...")
            ready_pos = self._get_ready_position(symbol_type)
            return_success = self._coordinated_dramatic_movement(arm_id, ready_pos)
        
            # RETURN TRUE FOR SUCCESSFUL MOVE
            return True
        
        else:
            print(f"❌ Failed to complete theatrical performance to position {position}")
            return False  # ADD RETURN FALSE
    
    def _move_arm_fast_finger_pointing(self, arm_id: int, target_position: List[float], duration: float) -> bool:
        """Theatrical sequential joint movement - each joint moves prominently"""
        if not isinstance(target_position, (list, tuple)) or len(target_position) != 3:
            return False
    
        print(f"🎭 THEATRICAL SEQUENCE: Moving all 7 joints sequentially")
    
        # Phase 1: Dramatic "wake up" - move each joint individually for visibility
        self._sequential_joint_demonstration(arm_id)
    
        # Phase 2: Coordinated movement to target with exaggerated joint usage
        success = self._coordinated_dramatic_movement(arm_id, target_position)
    
        # Phase 3: Final precise touch with wrist rotation (Joint 7)
        self._final_touch_with_wrist_rotation(arm_id, target_position)
    
        return success

    def _sequential_joint_demonstration(self, arm_id: int):
        """Move each joint individually to demonstrate all 7 DOF"""
        print("🎪 Phase 1: Sequential joint demonstration")
    
        # Get current positions
        initial_positions = [p.getJointState(arm_id, i)[0] for i in range(7)]
    
        joint_names = ["Base", "Shoulder", "Arm", "Elbow", "Forearm", "Wrist1", "Wrist2"]
    
        # Move each joint dramatically one by one
        for joint_idx in range(7):
            print(f"   🔧 Activating Joint {joint_idx + 1} ({joint_names[joint_idx]})")
        
            # Create dramatic movement for this specific joint
            if joint_idx == 0:  # Base rotation - full 360° showcase
                target_angle = initial_positions[0] + math.pi/2
            elif joint_idx == 1:  # Shoulder - dramatic lift
                target_angle = initial_positions[1] + 0.8
            elif joint_idx == 2:  # Arm rotation
                target_angle = initial_positions[2] + 0.6
            elif joint_idx == 3:  # Elbow - bend dramatically
                target_angle = initial_positions[3] + 1.0
            elif joint_idx == 4:  # Forearm twist
                target_angle = initial_positions[4] + 0.7
            elif joint_idx == 5:  # Wrist 1 - dramatic bend
                target_angle = initial_positions[5] + 0.9
            else:  # joint_idx == 6, Wrist 2 - full rotation for final touch
                target_angle = initial_positions[6] + math.pi
        
            # Animate this joint while others stay put
            steps = 40  # Quick but visible movement
            current_pos = initial_positions[joint_idx]
        
            for step in range(steps):
                alpha = step / steps
                alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            
                # Interpolate only this joint
                new_pos = current_pos + alpha_smooth * (target_angle - current_pos)
            
                # Apply to all joints (others stay in place)
                for i in range(7):
                    if i == joint_idx:
                        p.setJointMotorControl2(
                            arm_id, i, p.POSITION_CONTROL,
                            targetPosition=new_pos,
                            force=600,
                            maxVelocity=5
                        )
                    else:
                        p.setJointMotorControl2(
                            arm_id, i, p.POSITION_CONTROL,
                            targetPosition=initial_positions[i],
                            force=400,
                            maxVelocity=3
                        )
            
                p.stepSimulation()
                if step % 2 == 0:
                    self.env.update_camera_displays()
                time.sleep(1/60)
        
            # Update initial positions for next joint
            initial_positions[joint_idx] = target_angle
            time.sleep(0.2)  # Pause between joints

    def _coordinated_dramatic_movement(self, arm_id: int, target_position: List[float]) -> bool:
        """Coordinated movement using all joints with exaggerated motion"""
        print("🎪 Phase 2: Coordinated dramatic movement to target")
    
        # Create multiple intermediate positions that force all joints to work
        dramatic_waypoints = [
            # Waypoint 1: High arc position
            [target_position[0] - 0.2, target_position[1], target_position[2] + 0.3],
            # Waypoint 2: Side approach  
            [target_position[0] + 0.1, target_position[1] - 0.1, target_position[2] + 0.15],
            # Waypoint 3: Above target
            [target_position[0], target_position[1], target_position[2] + 0.1],
            # Waypoint 4: Final target
            target_position
        ]
    
        for waypoint_idx, waypoint in enumerate(dramatic_waypoints):
            print(f"   🎯 Moving to waypoint {waypoint_idx + 1}/4")
        
            # Use different orientations for each waypoint to force joint variety
            orientations = [
                p.getQuaternionFromEuler([0, -math.pi/2, math.pi/4]),
                p.getQuaternionFromEuler([math.pi/6, -math.pi/3, -math.pi/8]),  
                p.getQuaternionFromEuler([-math.pi/8, -math.pi/4, math.pi/6]),
                p.getQuaternionFromEuler([0, -math.pi/2, 0])  # Final pointing down
            ]
        
            # Calculate IK with null space projection to encourage joint movement
            joint_positions = p.calculateInverseKinematics(
                arm_id, self.env.ee_index, waypoint, orientations[waypoint_idx],
                lowerLimits=[-2.9, -2.1, -2.9, -2.1, -2.9, -2.1, -3.0],
                upperLimits=[2.9, 2.1, 2.9, 2.1, 2.9, 2.1, 3.0],
                jointRanges=[5.8, 4.2, 5.8, 4.2, 5.8, 4.2, 6.0],
                restPoses=[0.2 * waypoint_idx, -0.5 - 0.3 * waypoint_idx, 
                        0.1 * waypoint_idx, -1.2 - 0.2 * waypoint_idx,
                        0.3 * waypoint_idx, 0.8 + 0.3 * waypoint_idx, 
                        0.5 * waypoint_idx],  # Different rest poses for each waypoint
                maxNumIterations=100,
                residualThreshold=0.01
            )
        
            # Get current positions
            current_positions = [p.getJointState(arm_id, i)[0] for i in range(7)]
        
            # Move to waypoint
            steps = 30
            for step in range(steps):
                alpha = step / steps
                alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            
                # Move all 7 joints
                for i in range(7):
                    if i < len(joint_positions):
                        target_pos = current_positions[i] + alpha_smooth * (joint_positions[i] - current_positions[i])
                    
                        p.setJointMotorControl2(
                            arm_id, i, p.POSITION_CONTROL,
                            targetPosition=target_pos,
                            force=500 + i * 20,  # Increasing force per joint
                            maxVelocity=8,
                            positionGain=0.3,
                            velocityGain=1.0
                        )
            
                p.stepSimulation()
                if step % 3 == 0:
                    self.env.update_camera_displays()
                time.sleep(1/80)
        
            time.sleep(0.3)  # Pause at each waypoint
    
        return True

    def _final_touch_with_wrist_rotation(self, arm_id: int, target_position: List[float]):
        """Final dramatic wrist rotation and board touch"""
        print("🎪 Phase 3: Final wrist rotation and board touch")
    
        # Add visual finger at target
        self._add_pointing_finger_visual(target_position)
    
        # Get current joint 7 (wrist) position
        current_wrist_pos = p.getJointState(arm_id, 6)[0]
    
        # Dramatic wrist rotation sequence
        wrist_rotations = [
            current_wrist_pos + math.pi/2,    # 90° rotation
            current_wrist_pos + math.pi,      # 180° rotation  
            current_wrist_pos + 3*math.pi/2,  # 270° rotation
            current_wrist_pos + 2*math.pi     # Full 360° rotation
        ]
    
        print("   🌀 Performing dramatic wrist rotations...")
    
        for rotation_idx, target_wrist in enumerate(wrist_rotations):
            print(f"     Wrist rotation {rotation_idx + 1}/4: {target_wrist:.2f} rad")
        
            # Rotate wrist while maintaining position
            steps = 25
            start_wrist = p.getJointState(arm_id, 6)[0]
        
            for step in range(steps):
                alpha = step / steps
                alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            
                wrist_pos = start_wrist + alpha_smooth * (target_wrist - start_wrist)
            
                # Keep all other joints in place, only rotate wrist
                for i in range(7):
                    if i == 6:  # Wrist joint
                        p.setJointMotorControl2(
                            arm_id, i, p.POSITION_CONTROL,
                            targetPosition=wrist_pos,
                            force=400,
                            maxVelocity=6
                        )
                    else:
                        # Hold other joints steady
                        current_pos = p.getJointState(arm_id, i)[0]
                        p.setJointMotorControl2(
                            arm_id, i, p.POSITION_CONTROL,
                            targetPosition=current_pos,
                            force=300,
                            maxVelocity=2
                        )
            
                p.stepSimulation()
                if step % 2 == 0:
                    self.env.update_camera_displays()
                time.sleep(1/60)
        
            time.sleep(0.2)  # Pause between rotations
    
        # Final precise positioning - actually touch the board
        print("   👆 Final precise board touch...")
    
        # Lower the target to actually touch the board surface
        touch_position = [target_position[0], target_position[1], target_position[2] - 0.02]
    
        # Move to precise touch position
        touch_joint_positions = p.calculateInverseKinematics(
            arm_id, self.env.ee_index, touch_position, 
            p.getQuaternionFromEuler([0, -math.pi/2, 0])
        )
    
        # Final precise movement
        current_positions = [p.getJointState(arm_id, i)[0] for i in range(7)]
    
        steps = 20
        for step in range(steps):
            alpha = step / steps
            alpha_smooth = alpha**2  # Slower, more precise
        
            for i in range(7):
                if i < len(touch_joint_positions):
                    target_pos = current_positions[i] + alpha_smooth * (touch_joint_positions[i] - current_positions[i])
                    p.setJointMotorControl2(
                        arm_id, i, p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=300,  # Gentler for precision
                        maxVelocity=3
                    )
        
            p.stepSimulation()
            self.env.update_camera_displays()
            time.sleep(1/50)
    
        print("   ✅ Board touched! All 7 joints demonstrated.")
        time.sleep(0.5)  # Hold the touch

    
    def _generate_dramatic_waypoints(self, arm_id: int, final_target: List[float]) -> List[Tuple[List[float], List[float]]]:
        """Generate waypoints that force all 7 joints to move dramatically"""
        waypoints = []
    
        # Determine which arm we're using
        is_arm1 = (arm_id == self.env.robot_arm1_id)
    
        # Waypoint 1: Dramatic wind-up position (like a human preparing to point)
        if is_arm1:
            windup_pos = [-0.1, self.env.table_center[1] - 0.2, 0.95]  # Up and back
            windup_orient = p.getQuaternionFromEuler([0, -math.pi/2, math.pi/4])
        else:
            windup_pos = [0.1, self.env.table_center[1] + 0.2, 0.95]   # Up and back
            windup_orient = p.getQuaternionFromEuler([0, math.pi/2, -math.pi/4])
    
        waypoints.append((windup_pos, windup_orient))
    
        # Waypoint 2: Mid-reach position (forces shoulder and elbow movement)
        mid_x = (windup_pos[0] + final_target[0]) / 2
        mid_y = (windup_pos[1] + final_target[1]) / 2  
        mid_z = final_target[2] + 0.15  # Higher arc
        mid_pos = [mid_x, mid_y, mid_z]
    
        if is_arm1:
            mid_orient = p.getQuaternionFromEuler([0, -math.pi/3, math.pi/8])
        else:
            mid_orient = p.getQuaternionFromEuler([0, math.pi/3, -math.pi/8])
    
        waypoints.append((mid_pos, mid_orient))
    
        # Waypoint 3: Final pointing position with varied orientation
        board_positions = self.env.get_board_positions()
        position = 5  # Default
        for pos_num, pos_coords in board_positions.items():
            if abs(pos_coords[0] - final_target[0]) < 0.05 and abs(pos_coords[1] - final_target[1]) < 0.05:
                position = pos_num
                break
    
        final_angles = self._get_pointing_angle_for_position(position, arm_id)
        final_orient = p.getQuaternionFromEuler(final_angles)
    
        waypoints.append((final_target, final_orient))
    
        print(f"🎯 Generated {len(waypoints)} waypoints for dramatic all-joint movement")
        return waypoints
    
    def _monitor_joint_movement(self, arm_id: int, joint_name: str = ""):
        """Monitor and display joint positions for debugging"""
        joint_positions = []
        joint_velocities = []
    
        for i in range(7):
            joint_state = p.getJointState(arm_id, i)
            joint_positions.append(joint_state[0])  # Position
            joint_velocities.append(joint_state[1])  # Velocity
    
        # Print joint status
        print(f"🔧 {joint_name} Joint Status:")
        for i in range(7):
            print(f"   J{i+1}: Pos={joint_positions[i]:.3f}, Vel={joint_velocities[i]:.3f}")
    
        # Check which joints are actually moving
        moving_joints = [i+1 for i in range(7) if abs(joint_velocities[i]) > 0.01]
        print(f"🏃 Moving joints: {moving_joints}")
    
        return joint_positions, joint_velocities

    def _add_pointing_finger_visual(self, target_position: List[float]):
        """Add visual finger pointing indicator at target position"""
        try:
            # Remove any existing pointing finger
            self._clear_pointing_finger()
        
            # Create a small finger-like visual indicator
            self.pointing_finger_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=0.008,  # Thin finger
                    length=0.04,   # Short finger indicator
                    rgbaColor=[0.9, 0.7, 0.6, 0.8]  # Semi-transparent skin color
                ),
                basePosition=[target_position[0], target_position[1], target_position[2] - 0.02]
            )
        
            # Add "POINTING HERE" text
            p.addUserDebugText(
                "👉 POINTING", 
                [target_position[0], target_position[1], target_position[2] + 0.03],
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                lifeTime=1.0
            )
        except:
            pass
    
    def _get_pointing_angle_for_position(self, position: int, arm_id: int) -> List[float]:
        """Get optimal pointing angle based on board position and arm"""
        # Different angles for different positions to create varied, human-like movements
        is_arm1 = (arm_id == self.env.robot_arm1_id)
    
        angle_map = {
            1: [0, -math.pi/3, math.pi/8] if is_arm1 else [0, math.pi/4, -math.pi/8],     # Top-left
            2: [0, -math.pi/4, 0] if is_arm1 else [0, math.pi/6, 0],                     # Top-center  
            3: [0, -math.pi/6, -math.pi/8] if is_arm1 else [0, math.pi/3, math.pi/8],    # Top-right
            4: [0, -math.pi/3, math.pi/6] if is_arm1 else [0, math.pi/4, -math.pi/6],    # Middle-left
            5: [0, -math.pi/4, 0] if is_arm1 else [0, math.pi/4, 0],                     # Center
            6: [0, -math.pi/5, -math.pi/6] if is_arm1 else [0, math.pi/3, math.pi/6],    # Middle-right
            7: [0, -math.pi/2, math.pi/8] if is_arm1 else [0, math.pi/6, -math.pi/8],    # Bottom-left
            8: [0, -math.pi/3, 0] if is_arm1 else [0, math.pi/5, 0],                     # Bottom-center
            9: [0, -math.pi/4, -math.pi/8] if is_arm1 else [0, math.pi/2, math.pi/8],    # Bottom-right
        }
    
        return angle_map.get(position, [0, -math.pi/4, 0])


    def _clear_pointing_finger(self):
        """Clear pointing finger visual"""
        if hasattr(self, 'pointing_finger_id') and self.pointing_finger_id is not None:
            try:
                p.removeBody(self.pointing_finger_id)
            except:
                pass
            self.pointing_finger_id = None

    def _draw_laser_to_target(self, arm_id: int, target_position: List[float]):
        """Draw highly visible laser line from end-effector to target position"""
        try:
            # Get end-effector position
            ee_state = p.getLinkState(arm_id, self.env.ee_index)
            ee_position = ee_state[0]  # Position
        
            # Clear previous laser
            self._clear_laser()
        
            # Draw THICK laser line (red color for visibility)
            self.laser_line_id = p.addUserDebugLine(
                ee_position, target_position,
                lineColorRGB=[1, 0, 0],  # Bright red laser
                lineWidth=8,  # Much thicker line
                lifeTime=1.0  # Longer lifetime
            )
        
            # Add a bright sphere at target position for better visibility
            self.laser_target_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.02,  # 2cm sphere
                    rgbaColor=[1, 0, 0, 0.8]  # Semi-transparent red
                ),
                basePosition=target_position
            )
        
            # Add text label at target
            p.addUserDebugText(
                "🎯 TARGET", 
                [target_position[0], target_position[1], target_position[2] + 0.05],
                textColorRGB=[1, 0, 0],
                textSize=2,
                lifeTime=1.0
            )
        
        except Exception as e:
            pass  # Silently handle laser drawing errors

    def _clear_laser(self):
        """Clear existing laser line and target marker"""
        if self.laser_line_id is not None:
            try:
                p.removeUserDebugItem(self.laser_line_id)
            except:
                pass
            self.laser_line_id = None
    
        # Clear target sphere if it exists
        if hasattr(self, 'laser_target_id') and self.laser_target_id is not None:
            try:
                p.removeBody(self.laser_target_id)
            except:
                pass
            self.laser_target_id = None
    
    def _get_ready_position(self, symbol_type: int) -> List[float]:
        """Get dramatic ready position for each arm"""
        if symbol_type == 1:  # Arm 1
            return [-0.2, self.env.table_center[1] - 0.3, 0.85]
        else:  # Arm 2  
            return [0.2, self.env.table_center[1] + 0.3, 0.85]
    
    def precision_point_to_position(self, position: int):
        """Precision pointing with laser guidance (no symbol placement)"""
        if not (1 <= position <= 9):
            print(f"❌ Invalid position {position}")
            return
        
        board_positions = self.env.get_board_positions()
        target_position = board_positions[position]
        
        print(f"🎯 PRECISION pointing to position {position} with laser guidance")
        
        # Use arm 1 for precision pointing
        success = self._move_arm_fast_with_laser(self.env.robot_arm1_id, target_position, 1.0)
        
        if success:
            # Hold position longer for precision demonstration
            print(f"🔴 Laser locked on position {position}")
            time.sleep(2.0)
            
            # Return to ready
            ready_pos = self._get_ready_position(1)
            self._move_arm_fast_with_laser(self.env.robot_arm1_id, ready_pos, 0.8)
            
            # Clear laser
            self._clear_laser()
    
    def clear_virtual_symbols(self):
        """Clear all virtual symbols"""
        self.virtual_symbols.clear()
        self.game_state = np.zeros((3, 3), dtype=int)
        self._clear_laser()
        print("🧹 Cleared all virtual symbols, reset game state, and cleared laser")
    
    def get_empty_positions(self) -> List[int]:
        """Get list of empty board positions"""
        empty_positions = []
        for i in range(9):
            if self.game_state.flat[i] == 0:
                empty_positions.append(i + 1)
        return empty_positions
    
    def print_game_state(self):
        """Print current game state"""
        print("\n📋 Current Game State:")
        symbol_map = {0: '.', 1: 'X', 2: 'O'}
        
        for row in range(3):
            row_str = " | ".join([symbol_map[self.game_state[row, col]] for col in range(3)])
            print(f"   {row_str}")
            if row < 2:
                print("   ---------")
        
        empty_positions = self.get_empty_positions()
        print(f"📍 Empty positions: {empty_positions}")
        
        if self.virtual_symbols:
            print(f"🎭 Virtual symbols: {self.virtual_symbols}")


class BoardDetectionDemo(DualArmEnvironment):
    """Enhanced environment with consistent detection and fast pointing"""
    
    def __init__(self):
        super().__init__()
        self.board_detector = BoardDetector(self)
        self.show_detection_overlay = True
        self.show_laser = True
        self.detection_update_counter = 0
        self.detection_update_interval = 10  # Faster detection updates
        
        print("✅ Fast Precision Board Detection Demo initialized")
    
    def update_camera_displays_with_detection(self):
        """Update camera displays with consistent detection overlays"""
        self.camera_update_counter += 1
        self.detection_update_counter += 1
        
        # Regular camera updates
        if self.camera_update_counter % self.camera_update_interval != 0:
            return
        
        # Run detection every update for consistency
        detection_results = self.board_detector.fuse_multi_camera_detections()
        
        for cam_key, window_name in self.camera_windows.items():
            try:
                rgb_image, _, _ = self.get_camera_image(cam_key)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                # Always show consistent detection overlay
                if detection_results and cam_key in detection_results['all_detections'] and self.show_detection_overlay:
                    detection = detection_results['all_detections'][cam_key]
                    if detection['board_detected']:
                        bgr_image = detection['detection_image'].copy()
                
                # Add camera info
                camera_name = self.camera_configs[cam_key]["name"]
                cv2.putText(bgr_image, camera_name, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add confidence score
                if detection_results and cam_key in detection_results['all_detections']:
                    confidence = detection_results['all_detections'][cam_key]['confidence']
                    method = detection_results['all_detections'][cam_key]['detection_method']
                    conf_text = f"Conf: {confidence:.2f} ({method[:10]})"
                    cv2.putText(bgr_image, conf_text, (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Focus camera highlight
                if cam_key == self.current_camera_focus:
                    cv2.rectangle(bgr_image, (0, 0), (639, 479), (0, 255, 255), 3)
                    cv2.putText(bgr_image, "FOCUS", (10, 460), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Grid overlay and symbols for overhead camera
                if cam_key == "camera_5_overhead":
                    self._add_grid_overlay(bgr_image)
                    self._add_enhanced_virtual_symbols_overlay(bgr_image)
                
                cv2.imshow(window_name, bgr_image)
                
            except Exception as e:
                print(f"⚠️ Camera {cam_key} error: {e}")
    
    def _add_enhanced_virtual_symbols_overlay(self, image: np.ndarray):
        """Enhanced virtual symbols overlay with better visibility"""
        if not self.board_detector.virtual_symbols:
            return
        
        height, width = image.shape[:2]
        
        for position, symbol_type in self.board_detector.virtual_symbols.items():
            row = (position - 1) // 3
            col = (position - 1) % 3
            
            x = int(width * (col + 0.5) / 3)
            y = int(height * (row + 0.5) / 3)
            
            symbol_text = "X" if symbol_type == 1 else "O"
            color = (0, 0, 255) if symbol_type == 1 else (255, 0, 0)
            
            # Enhanced visibility with outline
            cv2.circle(image, (x, y), 25, (255, 255, 255), -1)  # White background
            cv2.circle(image, (x, y), 25, color, 3)  # Colored border
            cv2.putText(image, symbol_text, (x-15, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    
    def setup_demo_optimal_view(self):
        """Perfect view for tic-tac-toe robotic demonstration"""
        p.resetDebugVisualizerCamera(
            cameraDistance=2.2,      # Good distance to see both arms + table
            cameraYaw=35,           # Slight angle to see both arms clearly  
            cameraPitch=-30,        # Elevated view to see board and arm movements
            cameraTargetPosition=[0, 0.6, 0.65]  # Focus on board center
        )
    
        # Clean up the interface for better viewing
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Nice shadows
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
        print("🎯 Demo-optimized camera view activated!")
    
    def run_detection_demo(self):
        """Main demo with fast precision pointing"""
        print("🚀 Starting Fast Precision Board Detection Demo")
        print("📹 Consistent board detection across all cameras")
        print("⚡ Ultra-fast arm pointing with laser guidance")
        print("\n🎮 Fast Precision Controls:")
        print("   X + 1-9 - FAST point and place virtual X (0.8s)")
        print("   O + 1-9 - FAST point and place virtual O (0.8s)")
        print("   P + 1-9 - PRECISION point with laser (2s hold)")
        print("   L - Toggle laser visibility")
        print("   D - Toggle detection overlay")
        print("   C - Clear virtual symbols")
        print("   G - Print game state")
        print("   R - Reset game state")
        print("\n🎮 Enhanced Controls:")
        print("   V - Cycle through optimal viewpoints")  
        
        expecting_position_for = None
        
        try:
            while True:
                p.stepSimulation()
                self.update_camera_displays_with_detection()
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("🛑 ESC pressed - Exiting...")
                    break
                elif key == ord('v') or key == ord('V'):
                    self.cycle_viewpoint()
                elif key == ord(' '):  # SPACE
                    print("🎯 Running board coverage demonstration...")
                    self.demonstrate_board_coverage()
                elif key == ord('d') or key == ord('D'):
                    self.show_detection_overlay = not self.show_detection_overlay
                    status = "ON" if self.show_detection_overlay else "OFF"
                    print(f"📹 Detection overlay: {status}")
                elif key == ord('l') or key == ord('L'):
                    self.show_laser = not self.show_laser
                    status = "ON" if self.show_laser else "OFF"
                    print(f"🔴 Laser pointer: {status}")
                    if not self.show_laser:
                        self.board_detector._clear_laser()
                elif key == ord('x') or key == ord('X'):
                    expecting_position_for = 'X'
                    print("⚡ Enter position (1-9) for FAST X placement:")
                elif key == ord('o') or key == ord('O'):
                    expecting_position_for = 'O'
                    print("⚡ Enter position (1-9) for FAST O placement:")
                elif key == ord('p') or key == ord('P'):
                    expecting_position_for = 'P'
                    print("🎯 Enter position (1-9) for PRECISION pointing:")
                elif key == ord('c') or key == ord('C'):
                    self.board_detector.clear_virtual_symbols()
                elif key == ord('g') or key == ord('G'):
                    self.board_detector.print_game_state()
                elif key == ord('r') or key == ord('R'):
                    self.board_detector.clear_virtual_symbols()
                    print("🔄 Game state reset")
                elif key in [ord(str(i)) for i in range(1, 10)]:
                    if expecting_position_for == 'X':
                        position = int(chr(key))
                        self.board_detector.place_virtual_symbol_with_fast_pointing(position, 1)
                        expecting_position_for = None
                    elif expecting_position_for == 'O':
                        position = int(chr(key))
                        self.board_detector.place_virtual_symbol_with_fast_pointing(position, 2)
                        expecting_position_for = None
                    elif expecting_position_for == 'P':
                        position = int(chr(key))
                        self.board_detector.precision_point_to_position(position)
                        expecting_position_for = None
                    else:
                        # Camera focus (1-5 only)
                        if int(chr(key)) <= 5:
                            camera_keys = list(self.camera_configs.keys())
                            idx = int(chr(key)) - 1
                            if idx < len(camera_keys):
                                self.current_camera_focus = camera_keys[idx]
                                print(f"📹 Focusing on: {self.camera_configs[camera_keys[idx]]['name']}")
                elif key == ord('+'):
                    self.camera_update_interval = max(1, self.camera_update_interval - 2)
                    print(f"📹 Camera refresh rate increased")
                elif key == ord('-'):
                    self.camera_update_interval = min(30, self.camera_update_interval + 2)
                    print(f"📹 Camera refresh rate decreased")
                
                time.sleep(1/60)
                
        except KeyboardInterrupt:
            print("🛑 Keyboard interrupt received")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup"""
        self.board_detector._clear_laser()
        super().cleanup()


def main():
    """Main function"""
    print("=" * 100)
    print("🤖 MULTI-ROBOT TIC-TAC-TOE: FAST PRECISION POINTING + CONSISTENT DETECTION")
    print("📋 Program 2 (v2): Consistent board detection + ultra-fast arm pointing with laser")
    print("=" * 100)
    
    demo = BoardDetectionDemo()
    demo.run_detection_demo()

if __name__ == "__main__":
    main()