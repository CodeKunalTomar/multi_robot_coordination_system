#!/usr/bin/env python3
"""
Multi-Robot Tic-Tac-Toe: Computer Vision Board Detection System (REVISED)
Program 2: Fixed detection, proper game state tracking, and arm pointing
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
        self.board_size_pixels = {}  # Store detected board size for each camera
        self.board_corners = {}      # Store detected board corners for each camera
        self.game_state = np.zeros((3, 3), dtype=int)  # 0=empty, 1=X, 2=O
        self.confidence_threshold = 0.3  # Lowered for virtual environment
        
        # Detection parameters optimized for PyBullet
        self.detection_history = []  
        self.history_size = 5
        
        # Symbol templates
        self.x_template = None
        self.o_template = None
        self._create_symbol_templates()
        
        # Virtual symbols for testing (simulated X and O placements)
        self.virtual_symbols = {}  # {position: symbol_type}
        
        # Board detection cache for virtual environment
        self.virtual_board_detected = True  # Assume board is always detectable
        self.virtual_board_corners = self._create_virtual_board_corners()
        
        print("✅ Board Detection System initialized (Virtual Environment Mode)")
        self._print_controls()
    
    def _print_controls(self):
        """Print controls for board detection"""
        print("\n🎮 Board Detection Controls:")
        print("   D - Toggle detection overlay display")
        print("   X + 1-9 - Point arm and place virtual X at position")
        print("   O + 1-9 - Point arm and place virtual O at position")
        print("   C - Clear all virtual symbols")
        print("   G - Print current game state")
        print("   R - Reset game state")
        print("   P + 1-9 - Point arm to position without placing symbol")
    
    def _create_symbol_templates(self):
        """Create template images for X and O symbol detection"""
        # Create larger, more distinctive templates for virtual environment
        x_template = np.zeros((40, 40), dtype=np.uint8)
        cv2.line(x_template, (8, 8), (32, 32), 255, 4)
        cv2.line(x_template, (32, 8), (8, 32), 255, 4)
        self.x_template = x_template
        
        o_template = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(o_template, (20, 20), 12, 255, 4)
        self.o_template = o_template
        
        print("✅ Symbol templates created (Optimized for virtual environment)")
    
    def _create_virtual_board_corners(self):
        """Create virtual board corners for each camera when physical detection fails"""
        virtual_corners = {}
        
        for cam_key in self.env.camera_configs.keys():
            if cam_key == "camera_5_overhead":
                # Overhead camera gets full board view
                virtual_corners[cam_key] = np.array([
                    [200, 150],  # Top-left
                    [440, 150],  # Top-right  
                    [440, 330],  # Bottom-right
                    [200, 330]   # Bottom-left
                ], dtype=np.float32)
            else:
                # Side cameras get perspective view
                virtual_corners[cam_key] = np.array([
                    [180, 200],  # Top-left
                    [460, 180],  # Top-right
                    [480, 350],  # Bottom-right
                    [160, 370]   # Bottom-left
                ], dtype=np.float32)
        
        return virtual_corners
    
    def detect_board_in_image(self, image: np.ndarray, camera_key: str) -> Dict[str, Any]:
        """Detect tic-tac-toe board in a single camera image (Enhanced for virtual environment)"""
        result = {
            'board_detected': False,
            'corners': None,
            'grid_cells': None,
            'confidence': 0.0,
            'detection_image': image.copy(),
            'detection_method': 'none'
        }
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Try enhanced edge detection for virtual environment
        board_corners = self._detect_board_enhanced(gray, image)
        
        if board_corners is not None:
            result['board_detected'] = True
            result['corners'] = board_corners
            result['confidence'] = 0.8
            result['detection_method'] = 'enhanced_edge'
        else:
            # Method 2: Use virtual board corners as fallback
            if camera_key in self.virtual_board_corners:
                result['board_detected'] = True
                result['corners'] = self.virtual_board_corners[camera_key]
                result['confidence'] = 0.6
                result['detection_method'] = 'virtual_fallback'
        
        # Extract grid cells if board detected
        if result['board_detected']:
            result['grid_cells'] = self._extract_grid_cells(image, result['corners'])
            # Draw detection overlay
            self._draw_detection_overlay(result['detection_image'], 
                                       result['corners'], 
                                       result['detection_method'],
                                       result['confidence'])
        
        return result
    
    def _detect_board_enhanced(self, gray: np.ndarray, color_image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced board detection optimized for PyBullet virtual environment"""
        # Method 1: Color-based detection (look for the gray board)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Define range for gray board color
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([255, 30, 255])
        
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (should be the board)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate to quadrilateral
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
            
            # If not exactly 4 points, use bounding rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            return box.astype(np.float32)
        
        # Method 2: Enhanced edge detection
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None and len(lines) >= 4:
            horizontal_lines, vertical_lines = self._classify_lines(lines)
            
            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                corners = self._find_board_corners(horizontal_lines, vertical_lines, gray.shape)
                if corners is not None:
                    return corners
        
        return None
    
    def _classify_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """Classify detected lines as horizontal or vertical"""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            rho, theta = line[0]
            angle_deg = theta * 180 / np.pi
            
            # More lenient angle classification for virtual environment
            if abs(angle_deg - 90) < 30 or abs(angle_deg - 270) < 30:  # Vertical
                vertical_lines.append((rho, theta))
            elif abs(angle_deg) < 30 or abs(angle_deg - 180) < 30:  # Horizontal
                horizontal_lines.append((rho, theta))
        
        return horizontal_lines, vertical_lines
    
    def _find_board_corners(self, h_lines: List, v_lines: List, img_shape: Tuple) -> Optional[np.ndarray]:
        """Find board corners from line intersections"""
        height, width = img_shape[:2]
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None
        
        # Sort lines by rho
        h_lines.sort(key=lambda x: abs(x[0]))
        v_lines.sort(key=lambda x: abs(x[0]))
        
        # Take lines that are most likely board boundaries
        top_line = min(h_lines, key=lambda x: abs(x[0]))
        bottom_line = max(h_lines, key=lambda x: abs(x[0]))
        left_line = min(v_lines, key=lambda x: abs(x[0]))
        right_line = max(v_lines, key=lambda x: abs(x[0]))
        
        # Find intersections
        corners = []
        line_pairs = [
            (top_line, left_line),    # Top-left
            (top_line, right_line),   # Top-right
            (bottom_line, right_line), # Bottom-right
            (bottom_line, left_line)   # Bottom-left
        ]
        
        for h_line, v_line in line_pairs:
            intersection = self._line_intersection(h_line, v_line)
            if intersection is not None:
                x, y = intersection
                if 0 <= x < width and 0 <= y < height:
                    corners.append([x, y])
        
        if len(corners) == 4:
            return np.array(corners, dtype=np.float32)
        
        return None
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
        """Find intersection point of two lines in polar form"""
        rho1, theta1 = line1
        rho2, theta2 = line2
        
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        c1, c2 = rho1, rho2
        
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-6:
            return None
        
        x = (c1 * b2 - c2 * b1) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        
        return (x, y)
    
    def _extract_grid_cells(self, image: np.ndarray, corners: np.ndarray) -> List[np.ndarray]:
        """Extract 9 grid cells from the detected board"""
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
        """Interpolate corner position using bilinear interpolation"""
        # corners: [top_left, top_right, bottom_right, bottom_left]
        top = corners[0] * (1-u) + corners[1] * u
        bottom = corners[3] * (1-u) + corners[2] * u
        return top * (1-v) + bottom * v
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Calculate angles from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        ordered = corners[sorted_indices]
        
        # Find top-left (minimum x + y)
        sums = ordered[:, 0] + ordered[:, 1]
        min_idx = np.argmin(sums)
        
        # Rotate to start with top-left
        ordered = np.roll(ordered, -min_idx, axis=0)
        
        return ordered
    
    def _extract_cell_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract a single cell using perspective transformation"""
        dst_size = 80  # Larger cell size for better template matching
        dst_points = np.array([
            [0, 0],
            [dst_size, 0],
            [dst_size, dst_size],
            [0, dst_size]
        ], dtype=np.float32)
        
        try:
            transform = cv2.getPerspectiveTransform(corners, dst_points)
            cell = cv2.warpPerspective(image, transform, (dst_size, dst_size))
            return cell
        except:
            # Return empty cell if transform fails
            return np.zeros((dst_size, dst_size, 3), dtype=np.uint8)
    
    def _draw_detection_overlay(self, image: np.ndarray, corners: np.ndarray, 
                              method: str, confidence: float):
        """Draw detection overlay on image"""
        if corners is not None:
            # Draw board corners with different colors based on detection method
            color = (0, 255, 0) if method == 'enhanced_edge' else (255, 0, 255)  # Green or Magenta
            
            for i, corner in enumerate(corners):
                cv2.circle(image, tuple(corner.astype(int)), 5, color, -1)
                cv2.putText(image, str(i), tuple(corner.astype(int) + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw board outline
            cv2.polylines(image, [corners.astype(int)], True, color, 2)
            
            # Draw grid lines
            self._draw_grid_lines(image, corners, color)
            
            # Add detection info
            info_text = f"{method}: {confidence:.2f}"
            cv2.putText(image, info_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_grid_lines(self, image: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int]):
        """Draw 3x3 grid lines on detected board"""
        corners = self._order_corners(corners)
        
        # Draw vertical grid lines
        for i in range(1, 3):
            p1 = self._interpolate_corner(corners, i/3, 0)
            p2 = self._interpolate_corner(corners, i/3, 1)
            cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 1)
        
        # Draw horizontal grid lines
        for i in range(1, 3):
            p1 = self._interpolate_corner(corners, 0, i/3)
            p2 = self._interpolate_corner(corners, 1, i/3)
            cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 1)
    
    def detect_symbols_in_cells(self, cells: List[np.ndarray]) -> List[int]:
        """Detect X and O symbols in extracted grid cells"""
        symbols = []
        
        for i, cell in enumerate(cells):
            pos = i + 1  # Position 1-9
            
            # Always check virtual symbols first (for testing)
            if pos in self.virtual_symbols:
                symbols.append(self.virtual_symbols[pos])
                continue
            
            if cell is None or cell.size == 0:
                symbols.append(0)  # Empty
                continue
            
            # Convert to grayscale
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
            
            # Template matching for X and O
            x_score = self._match_template(gray_cell, self.x_template)
            o_score = self._match_template(gray_cell, self.o_template)
            
            # Determine symbol based on scores
            if x_score > self.confidence_threshold and x_score > o_score:
                symbols.append(1)  # X
            elif o_score > self.confidence_threshold and o_score > x_score:
                symbols.append(2)  # O
            else:
                symbols.append(0)  # Empty
        
        return symbols
    
    def _match_template(self, image: np.ndarray, template: np.ndarray) -> float:
        """Match template against image and return confidence score"""
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            return 0.0
        
        try:
            # Resize template to match cell size if needed
            cell_size = min(image.shape[0], image.shape[1])
            if template.shape[0] != cell_size:
                template_resized = cv2.resize(template, (cell_size, cell_size))
            else:
                template_resized = template
            
            # Template matching
            result = cv2.matchTemplate(image, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            return max_val
        except:
            return 0.0
    
    def fuse_multi_camera_detections(self) -> Dict[str, Any]:
        """Fuse board detection results from all cameras"""
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
                continue
        
        # Update game state - FIXED: Always update from virtual symbols
        self._update_game_state_from_virtual_symbols()
        
        # If we have a good detection, also try to update from detected symbols
        if best_detection and best_detection['board_detected'] and best_detection['grid_cells']:
            detected_symbols = self.detect_symbols_in_cells(best_detection['grid_cells'])
            # Merge detected symbols with virtual symbols
            for i, symbol in enumerate(detected_symbols):
                pos = i + 1
                if pos not in self.virtual_symbols and symbol != 0:
                    row, col = divmod(i, 3)
                    self.game_state[row, col] = symbol
        
        return {
            'all_detections': all_detections,
            'best_detection': best_detection,
            'best_confidence': best_confidence,
            'game_state': self.game_state.copy(),
            'virtual_symbols': self.virtual_symbols.copy()
        }
    
    def _update_game_state_from_virtual_symbols(self):
        """Update game state matrix from virtual symbols"""
        # Reset game state
        self.game_state = np.zeros((3, 3), dtype=int)
        
        # Apply virtual symbols
        for position, symbol_type in self.virtual_symbols.items():
            if 1 <= position <= 9:
                row = (position - 1) // 3
                col = (position - 1) % 3
                self.game_state[row, col] = symbol_type
    
    def place_virtual_symbol_with_arm_pointing(self, position: int, symbol_type: int):
        """Place virtual symbol with robotic arm pointing"""
        if not (1 <= position <= 9 and symbol_type in [1, 2]):
            print(f"❌ Invalid position {position} or symbol type {symbol_type}")
            return
        
        # Get 3D position for the arm to point to
        board_positions = self.env.get_board_positions()
        target_position = board_positions[position]
        
        # Determine which arm should point (X = arm 1, O = arm 2)
        arm_id = self.env.robot_arm1_id if symbol_type == 1 else self.env.robot_arm2_id
        arm_name = "Arm 1 (X)" if symbol_type == 1 else "Arm 2 (O)"
        symbol_name = "X" if symbol_type == 1 else "O"
        
        print(f"🎯 {arm_name} pointing to position {position} for {symbol_name}")
        
        # Point to the position
        success = self.env.move_arm_to_position(arm_id, target_position, 2.0)
        
        if success:
            # Hold position for dramatic effect
            time.sleep(1.0)
            
            # Place the virtual symbol
            self.virtual_symbols[position] = symbol_type
            print(f"✅ Virtual {symbol_name} placed at position {position}")
            
            # Return arm to ready position
            if symbol_type == 1:  # Arm 1
                ready_pos = [0.3, self.env.table_center[1] - 0.3, 0.9]
            else:  # Arm 2
                ready_pos = [0.3, self.env.table_center[1] + 0.3, 0.9]
            
            self.env.move_arm_to_position(arm_id, ready_pos, 1.5)
        else:
            print(f"❌ Failed to point arm to position {position}")
    
    def point_arm_to_position(self, position: int):
        """Point arm to position without placing symbol (for testing)"""
        if not (1 <= position <= 9):
            print(f"❌ Invalid position {position}")
            return
        
        # Get 3D position
        board_positions = self.env.get_board_positions()
        target_position = board_positions[position]
        
        # Use arm 1 for pointing demonstration
        arm_id = self.env.robot_arm1_id
        
        print(f"🎯 Pointing to position {position}")
        
        # Point to the position
        success = self.env.move_arm_to_position(arm_id, target_position, 2.0)
        
        if success:
            time.sleep(1.0)  # Hold position
            
            # Return to ready position
            ready_pos = [0.3, self.env.table_center[1] - 0.3, 0.9]
            self.env.move_arm_to_position(arm_id, ready_pos, 1.5)
        else:
            print(f"❌ Failed to point to position {position}")
    
    def clear_virtual_symbols(self):
        """Clear all virtual symbols"""
        self.virtual_symbols.clear()
        self.game_state = np.zeros((3, 3), dtype=int)
        print("🧹 Cleared all virtual symbols and reset game state")
    
    def get_empty_positions(self) -> List[int]:
        """Get list of empty board positions (1-9)"""
        empty_positions = []
        for i in range(9):
            if self.game_state.flat[i] == 0:
                empty_positions.append(i + 1)
        return empty_positions
    
    def print_game_state(self):
        """Print current game state to console"""
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
    """Extended environment with enhanced board detection capabilities"""
    
    def __init__(self):
        super().__init__()
        self.board_detector = BoardDetector(self)
        self.show_detection_overlay = True
        self.detection_update_counter = 0
        self.detection_update_interval = 15  # Update detection every 15 frames
        
        print("✅ Enhanced Board Detection Demo initialized")
    
    def update_camera_displays_with_detection(self):
        """Update camera displays with enhanced board detection overlays"""
        self.camera_update_counter += 1
        self.detection_update_counter += 1
        
        # Regular camera updates
        if self.camera_update_counter % self.camera_update_interval != 0:
            return
        
        # Run board detection periodically
        run_detection = (self.detection_update_counter % self.detection_update_interval == 0)
        detection_results = None
        
        if run_detection:
            detection_results = self.board_detector.fuse_multi_camera_detections()
        
        for cam_key, window_name in self.camera_windows.items():
            try:
                rgb_image, _, _ = self.get_camera_image(cam_key)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                # Add detection overlay if available
                if (run_detection and detection_results and 
                    cam_key in detection_results['all_detections'] and 
                    self.show_detection_overlay):
                    
                    detection = detection_results['all_detections'][cam_key]
                    if detection['board_detected']:
                        bgr_image = detection['detection_image'].copy()
                
                # Add camera info and confidence (ALWAYS VISIBLE)
                camera_name = self.camera_configs[cam_key]["name"]
                cv2.putText(bgr_image, camera_name, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add confidence score (FIXED)
                if (detection_results and cam_key in detection_results['all_detections']):
                    confidence = detection_results['all_detections'][cam_key]['confidence']
                    method = detection_results['all_detections'][cam_key]['detection_method']
                    conf_text = f"Conf: {confidence:.2f} ({method})"
                    cv2.putText(bgr_image, conf_text, (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Highlight focus camera
                if cam_key == self.current_camera_focus:
                    cv2.rectangle(bgr_image, (0, 0), (639, 479), (0, 255, 255), 3)
                    cv2.putText(bgr_image, "FOCUS", (10, 460), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add grid overlay and virtual symbols for overhead camera
                if cam_key == "camera_5_overhead":
                    self._add_grid_overlay(bgr_image)
                    self._add_virtual_symbols_overlay(bgr_image)
                
                cv2.imshow(window_name, bgr_image)
                
            except Exception as e:
                print(f"⚠️ Camera {cam_key} error: {e}")
    
    def _add_virtual_symbols_overlay(self, image: np.ndarray):
        """Add virtual symbols overlay to overhead camera view"""
        if not self.board_detector.virtual_symbols:
            return
        
        height, width = image.shape[:2]
        
        for position, symbol_type in self.board_detector.virtual_symbols.items():
            # Calculate position in image
            row = (position - 1) // 3
            col = (position - 1) % 3
            
            x = int(width * (col + 0.5) / 3)
            y = int(height * (row + 0.5) / 3)
            
            # Draw symbol with better visibility
            symbol_text = "X" if symbol_type == 1 else "O"
            color = (0, 0, 255) if symbol_type == 1 else (255, 0, 0)  # Red X, Blue O
            
            # Draw background circle for better visibility
            cv2.circle(image, (x, y), 20, (255, 255, 255), -1)
            cv2.circle(image, (x, y), 20, color, 2)
            
            # Draw symbol
            cv2.putText(image, symbol_text, (x-12, y+8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    def run_detection_demo(self):
        """Main demo loop with enhanced board detection"""
        print("🚀 Starting Enhanced Board Detection Demo")
        print("📹 Enhanced board detection system with arm pointing active")
        print("\n🎮 Enhanced Controls:")
        print("   D - Toggle detection overlay")
        print("   X + 1-9 - Point arm and place virtual X")
        print("   O + 1-9 - Point arm and place virtual O")
        print("   P + 1-9 - Point arm to position (no symbol)")
        print("   C - Clear virtual symbols")
        print("   G - Print current game state") 
        print("   R - Reset game state")
        
        expecting_position_for = None  # Track if waiting for position input
        
        try:
            while True:
                # Update simulation
                p.stepSimulation()
                
                # Update camera displays with detection
                self.update_camera_displays_with_detection()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("🛑 ESC pressed - Exiting...")
                    break
                elif key == ord(' '):  # SPACE key
                    print("🎯 Running board coverage demonstration...")
                    self.demonstrate_board_coverage()
                elif key == ord('d') or key == ord('D'):
                    self.show_detection_overlay = not self.show_detection_overlay
                    status = "ON" if self.show_detection_overlay else "OFF"
                    print(f"📹 Detection overlay: {status}")
                elif key == ord('x') or key == ord('X'):
                    expecting_position_for = 'X'
                    print("🎯 Enter position (1-9) for X (arm will point first):")
                elif key == ord('o') or key == ord('O'):
                    expecting_position_for = 'O'
                    print("🎯 Enter position (1-9) for O (arm will point first):")
                elif key == ord('p') or key == ord('P'):
                    expecting_position_for = 'P'
                    print("🎯 Enter position (1-9) to point to:")
                elif key == ord('c') or key == ord('C'):
                    self.board_detector.clear_virtual_symbols()
                elif key == ord('g') or key == ord('G'):
                    self.board_detector.print_game_state()
                elif key == ord('r') or key == ord('R'):
                    self.board_detector.game_state = np.zeros((3, 3), dtype=int)
                    self.board_detector.clear_virtual_symbols()
                    print("🔄 Game state reset")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    if expecting_position_for == 'X':
                        position = int(chr(key))
                        self.board_detector.place_virtual_symbol_with_arm_pointing(position, 1)
                        expecting_position_for = None
                    elif expecting_position_for == 'O':
                        position = int(chr(key))
                        self.board_detector.place_virtual_symbol_with_arm_pointing(position, 2)
                        expecting_position_for = None
                    elif expecting_position_for == 'P':
                        position = int(chr(key))
                        self.board_detector.point_arm_to_position(position)
                        expecting_position_for = None
                    else:
                        # Focus camera (original functionality)
                        camera_keys = list(self.camera_configs.keys())
                        idx = int(chr(key)) - 1
                        if idx < len(camera_keys):
                            self.current_camera_focus = camera_keys[idx]
                            print(f"📹 Focusing on: {self.camera_configs[camera_keys[idx]]['name']}")
                elif key in [ord('6'), ord('7'), ord('8'), ord('9')]:
                    if expecting_position_for == 'X':
                        position = int(chr(key))
                        self.board_detector.place_virtual_symbol_with_arm_pointing(position, 1)
                        expecting_position_for = None
                    elif expecting_position_for == 'O':
                        position = int(chr(key))
                        self.board_detector.place_virtual_symbol_with_arm_pointing(position, 2)
                        expecting_position_for = None
                    elif expecting_position_for == 'P':
                        position = int(chr(key))
                        self.board_detector.point_arm_to_position(position)
                        expecting_position_for = None
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


def main():
    """Main function"""
    print("=" * 90)
    print("🤖 MULTI-ROBOT TIC-TAC-TOE: ENHANCED COMPUTER VISION + ARM POINTING")
    print("📋 Program 2 (Revised): Fixed detection, game state tracking, and arm pointing")
    print("=" * 90)
    
    # Create and run the enhanced detection demo
    demo = BoardDetectionDemo()
    demo.run_detection_demo()

if __name__ == "__main__":
    main()
