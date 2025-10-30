#!/usr/bin/env python3
"""
Multi-Robot Tic-Tac-Toe: Computer Vision Board Detection System
Program 2: Advanced board detection, game state recognition, and multi-camera fusion
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
        self.confidence_threshold = 0.7
        
        # Detection parameters
        self.detection_history = []  # Store last N detections for stability
        self.history_size = 5
        
        # Symbol templates (we'll create these dynamically)
        self.x_template = None
        self.o_template = None
        self._create_symbol_templates()
        
        # Virtual symbols for testing (simulated X and O placements)
        self.virtual_symbols = {}  # {position: symbol_type}
        
        print("✅ Board Detection System initialized")
        self._print_controls()
    
    def _print_controls(self):
        """Print additional controls for board detection"""
        print("\n🎮 Board Detection Controls (in addition to Program 1 controls):")
        print("   D - Toggle detection overlay display")
        print("   X - Place virtual X at position (1-9)")
        print("   O - Place virtual O at position (1-9)")
        print("   C - Clear all virtual symbols")
        print("   S - Save current board state detection")
        print("   R - Reset game state")
    
    def _create_symbol_templates(self):
        """Create template images for X and O symbol detection"""
        # Create X template (30x30 pixels)
        x_template = np.zeros((30, 30), dtype=np.uint8)
        cv2.line(x_template, (5, 5), (25, 25), 255, 3)
        cv2.line(x_template, (25, 5), (5, 25), 255, 3)
        self.x_template = x_template
        
        # Create O template (30x30 pixels)
        o_template = np.zeros((30, 30), dtype=np.uint8)
        cv2.circle(o_template, (15, 15), 10, 255, 3)
        self.o_template = o_template
        
        print("✅ Symbol templates created (X and O)")
    
    def detect_board_in_image(self, image: np.ndarray, camera_key: str) -> Dict[str, Any]:
        """Detect tic-tac-toe board in a single camera image"""
        result = {
            'board_detected': False,
            'corners': None,
            'grid_cells': None,
            'confidence': 0.0,
            'detection_image': image.copy()
        }
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        
        if lines is not None and len(lines) >= 4:
            # Process lines to find grid structure
            horizontal_lines, vertical_lines = self._classify_lines(lines)
            
            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                # Find board corners from line intersections
                corners = self._find_board_corners(horizontal_lines, vertical_lines, image.shape)
                
                if corners is not None:
                    result['board_detected'] = True
                    result['corners'] = corners
                    result['grid_cells'] = self._extract_grid_cells(image, corners)
                    result['confidence'] = self._calculate_detection_confidence(corners, image.shape)
                    
                    # Draw detection overlay
                    self._draw_detection_overlay(result['detection_image'], corners, horizontal_lines, vertical_lines)
        
        return result
    
    def _classify_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """Classify detected lines as horizontal or vertical"""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            rho, theta = line[0]
            
            # Convert to degrees
            angle_deg = theta * 180 / np.pi
            
            # Classify based on angle
            if abs(angle_deg - 90) < 20 or abs(angle_deg - 270) < 20:  # Vertical
                vertical_lines.append((rho, theta))
            elif abs(angle_deg) < 20 or abs(angle_deg - 180) < 20:  # Horizontal
                horizontal_lines.append((rho, theta))
        
        return horizontal_lines, vertical_lines
    
    def _find_board_corners(self, h_lines: List, v_lines: List, img_shape: Tuple) -> Optional[np.ndarray]:
        """Find board corners from line intersections"""
        height, width = img_shape[:2]
        
        # Sort lines
        h_lines.sort(key=lambda x: x[0])  # Sort by rho
        v_lines.sort(key=lambda x: x[0])
        
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            # Take extreme lines to form board boundary
            top_line = h_lines[0] if h_lines[0][0] > 0 else h_lines[-1]
            bottom_line = h_lines[-1] if h_lines[-1][0] > 0 else h_lines[0]
            left_line = v_lines[0] if v_lines[0][0] > 0 else v_lines[-1]
            right_line = v_lines[-1] if v_lines[-1][0] > 0 else v_lines[0]
            
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
        
        # Convert to Cartesian form: ax + by = c
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        c1, c2 = rho1, rho2
        
        # Solve system of equations
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-6:  # Lines are parallel
            return None
        
        x = (c1 * b2 - c2 * b1) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        
        return (x, y)
    
    def _extract_grid_cells(self, image: np.ndarray, corners: np.ndarray) -> List[np.ndarray]:
        """Extract 9 grid cells from the detected board"""
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)
        
        cells = []
        for row in range(3):
            for col in range(3):
                # Calculate cell corners
                top_left = corners[0] + (corners[1] - corners[0]) * col/3 + (corners[3] - corners[0]) * row/3
                top_right = corners[0] + (corners[1] - corners[0]) * (col+1)/3 + (corners[3] - corners[0]) * row/3
                bottom_left = corners[0] + (corners[1] - corners[0]) * col/3 + (corners[3] - corners[0]) * (row+1)/3
                bottom_right = corners[0] + (corners[1] - corners[0]) * (col+1)/3 + (corners[3] - corners[0]) * (row+1)/3
                
                cell_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                
                # Extract cell using perspective transform
                cell = self._extract_cell_perspective(image, cell_corners)
                cells.append(cell)
        
        return cells
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Sort by angle from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder to start from top-left
        ordered = corners[sorted_indices]
        
        # Find the actual top-left (minimum x + y)
        sums = ordered[:, 0] + ordered[:, 1]
        min_idx = np.argmin(sums)
        
        # Rotate array to start with top-left
        ordered = np.roll(ordered, -min_idx, axis=0)
        
        return ordered
    
    def _extract_cell_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract a single cell using perspective transformation"""
        # Define destination points (square cell)
        dst_size = 60
        dst_points = np.array([
            [0, 0],
            [dst_size, 0],
            [dst_size, dst_size],
            [0, dst_size]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        transform = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply transform
        cell = cv2.warpPerspective(image, transform, (dst_size, dst_size))
        
        return cell
    
    def _calculate_detection_confidence(self, corners: np.ndarray, img_shape: Tuple) -> float:
        """Calculate confidence score for board detection"""
        if corners is None:
            return 0.0
        
        # Check if corners form a reasonable quadrilateral
        area = cv2.contourArea(corners)
        hull_area = cv2.contourArea(cv2.convexHull(corners))
        
        if hull_area == 0:
            return 0.0
        
        # Confidence based on area ratio and corner positions
        area_ratio = area / hull_area
        
        # Check if corners are within image bounds
        height, width = img_shape[:2]
        in_bounds = all(0 <= corner[0] < width and 0 <= corner[1] < height for corner in corners)
        
        confidence = area_ratio * (1.0 if in_bounds else 0.5)
        
        return min(confidence, 1.0)
    
    def _draw_detection_overlay(self, image: np.ndarray, corners: np.ndarray, h_lines: List, v_lines: List):
        """Draw detection overlay on image"""
        if corners is not None:
            # Draw board corners
            for i, corner in enumerate(corners):
                cv2.circle(image, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
                cv2.putText(image, str(i), tuple(corner.astype(int) + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw board outline
            cv2.polylines(image, [corners.astype(int)], True, (0, 255, 0), 2)
            
            # Draw grid lines
            self._draw_grid_lines(image, corners)
    
    def _draw_grid_lines(self, image: np.ndarray, corners: np.ndarray):
        """Draw 3x3 grid lines on detected board"""
        corners = self._order_corners(corners)
        
        # Draw vertical grid lines
        for i in range(1, 3):
            p1 = corners[0] + (corners[1] - corners[0]) * i/3
            p2 = corners[3] + (corners[2] - corners[3]) * i/3
            cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 1)
        
        # Draw horizontal grid lines
        for i in range(1, 3):
            p1 = corners[0] + (corners[3] - corners[0]) * i/3
            p2 = corners[1] + (corners[2] - corners[1]) * i/3
            cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 1)
    
    def detect_symbols_in_cells(self, cells: List[np.ndarray]) -> List[int]:
        """Detect X and O symbols in extracted grid cells"""
        symbols = []
        
        for i, cell in enumerate(cells):
            if cell is None:
                symbols.append(0)  # Empty
                continue
            
            # Check for virtual symbols first (for testing)
            pos = i + 1
            if pos in self.virtual_symbols:
                symbols.append(self.virtual_symbols[pos])
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
        
        # Resize template to match cell size if needed
        cell_size = min(image.shape[0], image.shape[1])
        if template.shape[0] != cell_size:
            template = cv2.resize(template, (cell_size, cell_size))
        
        # Template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max_val
    
    def fuse_multi_camera_detections(self) -> Dict[str, Any]:
        """Fuse board detection results from all cameras"""
        all_detections = {}
        best_detection = None
        best_confidence = 0.0
        
        # Get detection from each camera
        for cam_key in self.env.camera_configs.keys():
            rgb_image, _, _ = self.env.get_camera_image(cam_key)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            detection = self.detect_board_in_image(bgr_image, cam_key)
            all_detections[cam_key] = detection
            
            if detection['confidence'] > best_confidence:
                best_confidence = detection['confidence']
                best_detection = detection
                best_detection['source_camera'] = cam_key
        
        # Update game state based on best detection
        if best_detection and best_detection['board_detected'] and best_detection['grid_cells']:
            symbols = self.detect_symbols_in_cells(best_detection['grid_cells'])
            self.game_state = np.array(symbols).reshape(3, 3)
        
        return {
            'all_detections': all_detections,
            'best_detection': best_detection,
            'best_confidence': best_confidence,
            'game_state': self.game_state.copy(),
            'virtual_symbols': self.virtual_symbols.copy()
        }
    
    def place_virtual_symbol(self, position: int, symbol_type: int):
        """Place a virtual symbol for testing (1=X, 2=O)"""
        if 1 <= position <= 9 and symbol_type in [1, 2]:
            self.virtual_symbols[position] = symbol_type
            symbol_name = "X" if symbol_type == 1 else "O"
            print(f"🎯 Placed virtual {symbol_name} at position {position}")
        else:
            print(f"❌ Invalid position {position} or symbol type {symbol_type}")
    
    def clear_virtual_symbols(self):
        """Clear all virtual symbols"""
        self.virtual_symbols.clear()
        print("🧹 Cleared all virtual symbols")
    
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
    """Extended environment with board detection capabilities"""
    
    def __init__(self):
        super().__init__()
        self.board_detector = BoardDetector(self)
        self.show_detection_overlay = True
        self.detection_update_counter = 0
        self.detection_update_interval = 30  # Update detection every 30 frames
        
        print("✅ Board Detection Demo initialized")
    
    def update_camera_displays_with_detection(self):
        """Update camera displays with board detection overlays"""
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
                        bgr_image = detection['detection_image']
                        
                        # Add confidence score
                        confidence_text = f"Confidence: {detection['confidence']:.2f}"
                        cv2.putText(bgr_image, confidence_text, (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Add camera info
                camera_name = self.camera_configs[cam_key]["name"]
                cv2.putText(bgr_image, camera_name, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Highlight focus camera
                if cam_key == self.current_camera_focus:
                    cv2.rectangle(bgr_image, (0, 0), (639, 479), (0, 255, 255), 3)
                    cv2.putText(bgr_image, "FOCUS", (10, 460), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add grid overlay for overhead camera
                if cam_key == "camera_5_overhead":
                    self._add_grid_overlay(bgr_image)
                    
                    # Add virtual symbols overlay
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
            
            # Draw symbol
            symbol_text = "X" if symbol_type == 1 else "O"
            color = (0, 0, 255) if symbol_type == 1 else (255, 0, 0)  # Red X, Blue O
            
            cv2.putText(image, symbol_text, (x-15, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    def run_detection_demo(self):
        """Main demo loop with board detection"""
        print("🚀 Starting Board Detection Demo")
        print("📹 Board detection system active with multi-camera fusion")
        print("\n🎮 Additional Controls:")
        print("   D - Toggle detection overlay")
        print("   X followed by 1-9 - Place virtual X at position")
        print("   O followed by 1-9 - Place virtual O at position") 
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
                    print("🎯 Enter position (1-9) for X:")
                elif key == ord('o') or key == ord('O'):
                    expecting_position_for = 'O'
                    print("🎯 Enter position (1-9) for O:")
                elif key == ord('c') or key == ord('C'):
                    self.board_detector.clear_virtual_symbols()
                elif key == ord('g') or key == ord('G'):
                    self.board_detector.print_game_state()
                elif key == ord('r') or key == ord('R'):
                    self.board_detector.game_state = np.zeros((3, 3), dtype=int)
                    self.board_detector.clear_virtual_symbols()
                    print("🔄 Game state reset")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    if expecting_position_for:
                        # Place virtual symbol
                        position = int(chr(key))
                        symbol_type = 1 if expecting_position_for == 'X' else 2
                        self.board_detector.place_virtual_symbol(position, symbol_type)
                        expecting_position_for = None
                    else:
                        # Focus camera (original functionality)
                        camera_keys = list(self.camera_configs.keys())
                        idx = int(chr(key)) - 1
                        if idx < len(camera_keys):
                            self.current_camera_focus = camera_keys[idx]
                            print(f"📹 Focusing on: {self.camera_configs[camera_keys[idx]]['name']}")
                elif key in [ord('6'), ord('7'), ord('8'), ord('9')]:
                    if expecting_position_for:
                        # Place virtual symbol
                        position = int(chr(key))
                        symbol_type = 1 if expecting_position_for == 'X' else 2
                        self.board_detector.place_virtual_symbol(position, symbol_type)
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
    print("=" * 80)
    print("🤖 MULTI-ROBOT TIC-TAC-TOE: COMPUTER VISION BOARD DETECTION")
    print("📋 Program 2: Advanced board detection and game state recognition")
    print("=" * 80)
    
    # Create and run the detection demo
    demo = BoardDetectionDemo()
    demo.run_detection_demo()

if __name__ == "__main__":
    main()
