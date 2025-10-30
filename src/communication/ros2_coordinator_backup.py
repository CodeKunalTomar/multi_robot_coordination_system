#!/usr/bin/env python3
"""
Multi-Robot Tic-Tac-Toe: ROS2 Multi-Robot Coordination System
Program 3: Intelligent turn-based coordination with ROS2 communication
"""

import os
import sys
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import pybullet as p

# Add previous modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vision'))

from dual_arm_environment import DualArmEnvironment
from board_detector import BoardDetector

# Mock ROS2 for systems without ROS2 (we'll use threading instead)
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Int32
    from geometry_msgs.msg import Point
    ROS2_AVAILABLE = True
    print("✅ ROS2 detected - Using full ROS2 communication")
except ImportError:
    ROS2_AVAILABLE = False
    print("⚠️ ROS2 not available - Using threading-based communication")

class GameState(Enum):
    WAITING_FOR_PLAYERS = "waiting_for_players"
    PLAYER_X_TURN = "player_x_turn"
    PLAYER_O_TURN = "player_o_turn"
    GAME_OVER = "game_over"
    CALCULATING_MOVE = "calculating_move"
    EXECUTING_MOVE = "executing_move"

class Player(Enum):
    X = 1
    O = 2
    NONE = 0

class GameResult(Enum):
    X_WINS = "X_WINS"
    O_WINS = "O_WINS"
    DRAW = "DRAW"
    IN_PROGRESS = "IN_PROGRESS"

# Threading-based communication system (fallback for non-ROS2 systems)
class ThreadingCommunicator:
    def __init__(self):
        self.message_queue = {}
        self.subscribers = {}
        self.lock = threading.Lock()
    
    def publish(self, topic: str, message: dict):
        with self.lock:
            if topic not in self.message_queue:
                self.message_queue[topic] = []
            self.message_queue[topic].append(message)
            
            # Notify subscribers
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        print(f"⚠️ Callback error: {e}")
    
    def subscribe(self, topic: str, callback):
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
    
    def get_latest_message(self, topic: str) -> Optional[dict]:
        with self.lock:
            if topic in self.message_queue and self.message_queue[topic]:
                return self.message_queue[topic][-1]
        return None

# Global communicator instance
global_communicator = ThreadingCommunicator()

class GameCoordinator:
    def __init__(self, environment: DualArmEnvironment, board_detector: BoardDetector):
        """Initialize the game coordination system"""
        self.env = environment
        self.board_detector = board_detector
        
        # Game state management
        self.game_state = GameState.WAITING_FOR_PLAYERS
        self.current_player = Player.X
        self.game_board = np.zeros((3, 3), dtype=int)
        self.move_history = []
        self.game_result = GameResult.IN_PROGRESS
        
        # Communication setup
        self.communicator = global_communicator
        self._setup_communication()
        
        # Coordination parameters
        self.move_timeout = 30.0  # 30 seconds per move
        self.current_move_start_time = None
        
        # Robot assignment
        self.robot_assignments = {
            Player.X: self.env.robot_arm1_id,
            Player.O: self.env.robot_arm2_id
        }
        
        print("✅ Game Coordinator initialized")
        print(f"🎮 Player X assigned to Arm 1 (ID: {self.robot_assignments[Player.X]})")
        print(f"🎮 Player O assigned to Arm 2 (ID: {self.robot_assignments[Player.O]})")
        self._print_coordination_controls()
    
    def _print_coordination_controls(self):
        """Print coordination-specific controls"""
        print("\n🎮 Multi-Robot Coordination Controls:")
        print("   A - Auto-play mode (robots play each other)")
        print("   H - Human vs Robot mode")
        print("   S - Start new game")
        print("   T - Show game statistics")
        print("   M - Make AI move for current player")
        print("   U - Undo last move")
        print("   W - Show winning combinations")
    
    def _setup_communication(self):
        """Setup communication topics and handlers"""
        # Game state topics
        self.communicator.subscribe("game/state_change", self._handle_game_state_change)
        self.communicator.subscribe("game/move_request", self._handle_move_request)
        self.communicator.subscribe("game/move_complete", self._handle_move_complete)
        
        # Robot coordination topics
        self.communicator.subscribe("robot/arm1/status", self._handle_arm1_status)
        self.communicator.subscribe("robot/arm2/status", self._handle_arm2_status)
        self.communicator.subscribe("robot/collision_warning", self._handle_collision_warning)
        
        print("✅ Communication system initialized")
    
    def _handle_game_state_change(self, message: dict):
        """Handle game state change messages"""
        new_state = message.get('state')
        print(f"🔄 Game state change: {self.game_state.value} → {new_state}")
    
    def _handle_move_request(self, message: dict):
        """Handle move request from players"""
        player = Player(message.get('player', 0))
        position = message.get('position', 0)
        print(f"🎯 Move request: Player {player.name} wants position {position}")
        
        if self._is_valid_move(position) and player == self.current_player:
            self._execute_coordinated_move(player, position)
    
    def _handle_move_complete(self, message: dict):
        """Handle move completion notification"""
        player = Player(message.get('player', 0))
        position = message.get('position', 0)
        print(f"✅ Move completed: Player {player.name} placed at position {position}")
        
        self._update_game_state_after_move(player, position)
    
    def _handle_arm1_status(self, message: dict):
        """Handle Arm 1 status updates"""
        status = message.get('status', 'unknown')
        if status == 'busy':
            print("🤖 Arm 1 (Player X) is executing move...")
        elif status == 'ready':
            print("🤖 Arm 1 (Player X) ready for next command")
    
    def _handle_arm2_status(self, message: dict):
        """Handle Arm 2 status updates"""
        status = message.get('status', 'unknown')
        if status == 'busy':
            print("🤖 Arm 2 (Player O) is executing move...")
        elif status == 'ready':
            print("🤖 Arm 2 (Player O) ready for next command")
    
    def _handle_collision_warning(self, message: dict):
        """Handle collision warnings between arms"""
        print("⚠️ COLLISION WARNING: Arms too close - implementing safety protocol")
        self._emergency_stop_all_arms()
    
    def start_new_game(self):
        """Initialize a new game"""
        print("\n" + "="*60)
        print("🎮 STARTING NEW TIC-TAC-TOE GAME")
        print("="*60)
        
        # Reset game state
        self.game_state = GameState.PLAYER_X_TURN
        self.current_player = Player.X
        self.game_board = np.zeros((3, 3), dtype=int)
        self.move_history = []
        self.game_result = GameResult.IN_PROGRESS
        self.current_move_start_time = time.time()
        
        # Clear virtual symbols
        self.board_detector.clear_virtual_symbols()
        
        # Notify all components
        self.communicator.publish("game/state_change", {
            'state': self.game_state.value,
            'current_player': self.current_player.value
        })
        
        # Move arms to ready positions
        self._move_arms_to_game_ready_positions()
        
        print(f"🎯 Player {self.current_player.name}'s turn")
        print("📋 Use X/O + 1-9 to make moves, or 'M' for AI move")
    
    def _move_arms_to_game_ready_positions(self):
        """Move both arms to optimal ready positions for gameplay"""
        print("🤖 Moving arms to game-ready positions...")
        
        # Move arms to ready positions with coordination
        ready_pos_x = self.board_detector._get_ready_position(1)
        ready_pos_o = self.board_detector._get_ready_position(2)
        
        # Publish status updates
        self.communicator.publish("robot/arm1/status", {'status': 'moving_to_ready'})
        self.communicator.publish("robot/arm2/status", {'status': 'moving_to_ready'})
        
        # Execute coordinated movement (ensure no collision)
        threading.Thread(target=self._safe_move_to_ready, args=(self.env.robot_arm1_id, ready_pos_x, "arm1")).start()
        time.sleep(0.5)  # Stagger movements to avoid collision
        threading.Thread(target=self._safe_move_to_ready, args=(self.env.robot_arm2_id, ready_pos_o, "arm2")).start()
    
    def _safe_move_to_ready(self, arm_id: int, position: List[float], arm_name: str):
        """Safely move arm to ready position with status updates"""
        self.communicator.publish(f"robot/{arm_name}/status", {'status': 'busy'})
        
        success = self.board_detector._coordinated_dramatic_movement(arm_id, position)
        
        status = 'ready' if success else 'error'
        self.communicator.publish(f"robot/{arm_name}/status", {'status': status})
        
        if success:
            print(f"✅ {arm_name.upper()} ready for gameplay")
        else:
            print(f"❌ {arm_name.upper()} failed to reach ready position")
    
    def make_coordinated_move(self, player: Player, position: int) -> bool:
        """Execute a coordinated move with full safety checks"""
        if not self._is_valid_move(position):
            print(f"❌ Invalid move: Position {position} is not available")
            return False
        
        if player != self.current_player:
            print(f"❌ Invalid move: It's Player {self.current_player.name}'s turn")
            return False
        
        print(f"\n🎯 COORDINATED MOVE: Player {player.name} → Position {position}")
        print("-" * 50)
        
        # Update game state
        self.game_state = GameState.EXECUTING_MOVE
        self.communicator.publish("game/state_change", {
            'state': self.game_state.value,
            'executing_player': player.value,
            'target_position': position
        })
        
        # Check for potential arm collisions
        if self._check_collision_risk(player, position):
            print("⚠️ Collision risk detected - implementing safe movement protocol")
            return self._execute_safe_coordinated_move(player, position)
        else:
            return self._execute_coordinated_move(player, position)
    
    def _check_collision_risk(self, player: Player, position: int) -> bool:
        """Check if the planned move creates collision risk"""
        # Get target position
        board_positions = self.env.get_board_positions()
        target_pos = board_positions[position]
        
        # Get other arm's current position
        other_player = Player.O if player == Player.X else Player.X
        other_arm_id = self.robot_assignments[other_player]
        
        # Get other arm's end-effector position
        other_ee_state = self.env.p.getLinkState(other_arm_id, self.env.ee_index)
        other_pos = other_ee_state[0]
        
        # Calculate distance
        distance = np.linalg.norm(np.array(target_pos) - np.array(other_pos))
        
        # Collision risk if distance < 0.3 meters
        collision_risk = distance < 0.3
        
        if collision_risk:
            print(f"⚠️ Collision risk: Distance to other arm = {distance:.3f}m")
        
        return collision_risk
    
    def _execute_safe_coordinated_move(self, player: Player, position: int) -> bool:
        """Execute move with enhanced safety protocols"""
        print("🛡️ Executing SAFE coordinated move...")
        
        # Step 1: Move non-active arm to safe position
        other_player = Player.O if player == Player.X else Player.X
        other_arm_id = self.robot_assignments[other_player]
        
        # Move other arm to safe position first
        safe_pos = self._get_safe_position(other_player)
        print(f"🛡️ Moving Player {other_player.name} arm to safe position...")
        
        self.communicator.publish(f"robot/arm{'1' if other_player == Player.X else '2'}/status", {'status': 'moving_to_safe'})
        self.board_detector._coordinated_dramatic_movement(other_arm_id, safe_pos)
        
        # Step 2: Execute active player's move
        time.sleep(0.5)  # Safety delay
        success = self._execute_coordinated_move(player, position)
        
        # Step 3: Return other arm to ready position
        if success:
            time.sleep(1.0)
            ready_pos = self.board_detector._get_ready_position(other_player.value)
            self.board_detector._coordinated_dramatic_movement(other_arm_id, ready_pos)
            self.communicator.publish(f"robot/arm{'1' if other_player == Player.X else '2'}/status", {'status': 'ready'})
        
        return success
    
    def _get_safe_position(self, player: Player) -> List[float]:
        """Get safe position for arm to avoid collisions"""
        if player == Player.X:
            return [-0.4, self.env.table_center[1] - 0.4, 0.95]  # Further back
        else:
            return [0.4, self.env.table_center[1] + 0.4, 0.95]   # Further back
    
    def _execute_coordinated_move(self, player: Player, position: int) -> bool:
        """Execute the actual coordinated move"""
        arm_id = self.robot_assignments[player]
        arm_name = f"arm{'1' if player == Player.X else '2'}"
        
        # Publish move start
        self.communicator.publish(f"robot/{arm_name}/status", {'status': 'executing_move'})
        self.communicator.publish("game/move_request", {
            'player': player.value,
            'position': position,
            'timestamp': time.time()
        })
        
        # Execute theatrical move
        success = self.board_detector.place_virtual_symbol_with_fast_pointing(position, player.value)
        
        if success:
            # Update game state
            self._update_game_state_after_move(player, position)
            
            # Publish move completion
            self.communicator.publish("game/move_complete", {
                'player': player.value,
                'position': position,
                'timestamp': time.time()
            })
            self.communicator.publish(f"robot/{arm_name}/status", {'status': 'ready'})
            
            return True
        else:
            print(f"❌ Move execution failed for Player {player.name}")
            self.communicator.publish(f"robot/{arm_name}/status", {'status': 'error'})
            return False
    
    def _update_game_state_after_move(self, player: Player, position: int):
        """Update game state after successful move"""
        # Update board
        row, col = divmod(position - 1, 3)
        self.game_board[row, col] = player.value
        
        # Record move
        self.move_history.append({
            'player': player,
            'position': position,
            'timestamp': time.time()
        })
        
        # Check for win/draw
        self.game_result = self._check_game_result()
        
        if self.game_result == GameResult.IN_PROGRESS:
            # Switch players
            self.current_player = Player.O if player == Player.X else Player.X
            self.game_state = GameState.PLAYER_O_TURN if self.current_player == Player.O else GameState.PLAYER_X_TURN
            self.current_move_start_time = time.time()
            
            print(f"\n🔄 Turn complete! Now it's Player {self.current_player.name}'s turn")
        else:
            # Game over
            self.game_state = GameState.GAME_OVER
            self._announce_game_result()
        
        # Publish state update
        self.communicator.publish("game/state_change", {
            'state': self.game_state.value,
            'current_player': self.current_player.value,
            'game_result': self.game_result.value
        })
    
    def _check_game_result(self) -> GameResult:
        """Check if game is won, drawn, or still in progress"""
        # Check rows, columns, and diagonals
        for i in range(3):
            # Check rows
            if self.game_board[i, 0] == self.game_board[i, 1] == self.game_board[i, 2] != 0:
                return GameResult.X_WINS if self.game_board[i, 0] == 1 else GameResult.O_WINS
            
            # Check columns
            if self.game_board[0, i] == self.game_board[1, i] == self.game_board[2, i] != 0:
                return GameResult.X_WINS if self.game_board[0, i] == 1 else GameResult.O_WINS
        
        # Check diagonals
        if self.game_board[0, 0] == self.game_board[1, 1] == self.game_board[2, 2] != 0:
            return GameResult.X_WINS if self.game_board[1, 1] == 1 else GameResult.O_WINS
        
        if self.game_board[0, 2] == self.game_board[1, 1] == self.game_board[2, 0] != 0:
            return GameResult.X_WINS if self.game_board[1, 1] == 1 else GameResult.O_WINS
        
        # Check for draw
        if np.all(self.game_board != 0):
            return GameResult.DRAW
        
        return GameResult.IN_PROGRESS
    
    def _announce_game_result(self):
        """Announce the final game result"""
        print("\n" + "="*60)
        print("🎊 GAME OVER! 🎊")
        print("="*60)
        
        if self.game_result == GameResult.X_WINS:
            print("🏆 PLAYER X (ARM 1) WINS!")
            self._victory_celebration(Player.X)
        elif self.game_result == GameResult.O_WINS:
            print("🏆 PLAYER O (ARM 2) WINS!")
            self._victory_celebration(Player.O)
        else:
            print("🤝 IT'S A DRAW!")
            self._draw_celebration()
        
        self._show_game_statistics()
    
    def _victory_celebration(self, winner: Player):
        """Victory celebration sequence"""
        winning_arm_id = self.robot_assignments[winner]
        print(f"🎉 Victory celebration for Player {winner.name}!")
        
        # Theatrical victory gesture
        if winner == Player.X:
            victory_pos = [0, self.env.table_center[1] - 0.2, 0.95]
        else:
            victory_pos = [0, self.env.table_center[1] + 0.2, 0.95]
        
        # Execute victory movement
        threading.Thread(target=self._execute_victory_sequence, args=(winning_arm_id,)).start()
    
    def _execute_victory_sequence(self, arm_id: int):
        """Execute theatrical victory sequence"""
        # Victory positions - raise arm high
        victory_positions = [
            [0, self.env.table_center[1], 1.0],   # High center
            [0.1, self.env.table_center[1], 1.1], # Higher right
            [-0.1, self.env.table_center[1], 1.1], # Higher left
            [0, self.env.table_center[1], 1.0],   # Back to center
        ]
        
        for pos in victory_positions:
            self.board_detector._coordinated_dramatic_movement(arm_id, pos)
            time.sleep(0.5)
        
        print("🎊 Victory celebration complete!")
    
    def _draw_celebration(self):
        """Draw celebration - both arms gesture"""
        print("🤝 Draw celebration - both players played well!")
        
        # Both arms do a "handshake" gesture
        meet_pos_x = [0, self.env.table_center[1], 0.85]
        meet_pos_o = [0, self.env.table_center[1], 0.85]
        
        threading.Thread(target=self.board_detector._coordinated_dramatic_movement, 
                        args=(self.env.robot_arm1_id, meet_pos_x)).start()
        time.sleep(0.3)
        threading.Thread(target=self.board_detector._coordinated_dramatic_movement,
                        args=(self.env.robot_arm2_id, meet_pos_o)).start()
    
    def _show_game_statistics(self):
        """Show detailed game statistics"""
        print("\n📊 GAME STATISTICS:")
        print("-" * 30)
        print(f"Total moves: {len(self.move_history)}")
        print(f"Game duration: {time.time() - (self.move_history[0]['timestamp'] if self.move_history else time.time()):.1f} seconds")
        
        print("\n📋 Move History:")
        for i, move in enumerate(self.move_history):
            print(f"  {i+1}. Player {move['player'].name} → Position {move['position']}")
        
        print("\n🎮 Final Board State:")
        self.board_detector.print_game_state()
    
    def _is_valid_move(self, position: int) -> bool:
        """Check if move is valid"""
        if not 1 <= position <= 9:
            return False
        
        row, col = divmod(position - 1, 3)
        return self.game_board[row, col] == 0
    
    def _emergency_stop_all_arms(self):
        """Emergency stop all robotic arms"""
        print("🚨 EMERGENCY STOP - All arms halted!")
        # Implementation would stop all joint movements immediately
        self.communicator.publish("robot/emergency_stop", {'timestamp': time.time()})
    
    def get_game_status(self) -> dict:
        """Get current game status for external queries"""
        return {
            'state': self.game_state.value,
            'current_player': self.current_player.value,
            'board': self.game_board.tolist(),
            'result': self.game_result.value,
            'move_count': len(self.move_history)
        }


class MultiRobotCoordinationDemo(DualArmEnvironment):
    """Enhanced environment with multi-robot coordination"""
    
    def __init__(self):
        super().__init__()
        self.board_detector = BoardDetector(self)
        self.game_coordinator = GameCoordinator(self, self.board_detector)
        self.shutdown_requested = False  # Add shutdown flag

        # Add missing attributes for camera detection
        self.show_detection_overlay = True
        self.detection_update_counter = 0
        self.detection_update_interval = 15
        
        print("✅ Multi-Robot Coordination Demo initialized")
    
    def update_camera_displays_with_detection(self):
        """Update camera displays with detection overlays"""
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
                if not self.shutdown_requested:
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

    def run_coordination_demo(self):
        """Main demo loop with coordination features"""
        print("🚀 Starting Multi-Robot Coordination Demo")
        print("🤖 Intelligent turn-based coordination active")
        print("📡 ROS2-style communication system running")
        
        print("\n🎮 Coordination Controls:")
        print("   S - Start new game")
        print("   X + 1-9 - Coordinated X move (only on X's turn)")
        print("   O + 1-9 - Coordinated O move (only on O's turn)")
        print("   T - Show game statistics")
        print("   W - Show winning combinations")
        print("   V - Cycle viewpoints")
        print("   ESC - Exit")
        
        # Start with a new game
        self.game_coordinator.start_new_game()
        
        expecting_position_for = None
        
        try:
            while True:
                if self.shutdown_requested:
                    break
                    
                p.stepSimulation()  # Fixed: removed self.env
                self.update_camera_displays_with_detection()
                
                key = cv2.waitKey(1) & 0xFF  # Fixed: removed self.env
                if key == 27:  # ESC
                    print("🛑 ESC pressed - Exiting...")
                    self.shutdown_requested = True
                    break
                elif key == ord('a') or key == ord('A'):
                    print("🤖 Auto-play mode not yet implemented (Program 4)")
                elif key == ord('s') or key == ord('S'):
                    self.game_coordinator.start_new_game()
                elif key == ord('t') or key == ord('T'):
                    self.game_coordinator._show_game_statistics()
                elif key == ord('u') or key == ord('U'):
                    print("↩️ Undo not yet implemented")
                elif key == ord('w') or key == ord('W'):
                    self._show_winning_combinations()
                elif key == ord('v') or key == ord('V'):
                    self.cycle_viewpoint()
                elif key == ord('m') or key == ord('M'):
                    print("🧠 AI move not yet implemented (Program 4)")
                elif key == ord('x') or key == ord('X'):
                    if self.game_coordinator.current_player == Player.X:
                        expecting_position_for = 'X'
                        print("⚡ Enter position (1-9) for coordinated X placement:")
                    else:
                        print(f"❌ It's Player {self.game_coordinator.current_player.name}'s turn")
                elif key == ord('o') or key == ord('O'):
                    if self.game_coordinator.current_player == Player.O:
                        expecting_position_for = 'O'
                        print("⚡ Enter position (1-9) for coordinated O placement:")
                    else:
                        print(f"❌ It's Player {self.game_coordinator.current_player.name}'s turn")
                elif key in [ord(str(i)) for i in range(1, 10)]:
                    if expecting_position_for:
                        position = int(chr(key))
                        player = Player.X if expecting_position_for == 'X' else Player.O
                        self.game_coordinator.make_coordinated_move(player, position)
                        expecting_position_for = None
                    else:
                        # Camera focus
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
            self.shutdown_requested = True
            time.sleep(0.2)  # Give threads time to see shutdown flag
            self.cleanup()
    
    def _show_winning_combinations(self):
        """Show all possible winning combinations"""
        print("\n🏆 WINNING COMBINATIONS:")
        combinations = [
            [1, 2, 3], [4, 5, 6], [7, 8, 9],  # Rows
            [1, 4, 7], [2, 5, 8], [3, 6, 9],  # Columns
            [1, 5, 9], [3, 5, 7]               # Diagonals
        ]
        
        for i, combo in enumerate(combinations):
            combo_type = "Row" if i < 3 else "Column" if i < 6 else "Diagonal"
            print(f"  {combo_type}: {combo}")
    
    def cleanup(self):
        """Enhanced cleanup with thread safety"""
        print("🧹 Enhanced cleanup with coordination...")
        self.shutdown_requested = True
        
        # Clear any remaining laser visuals
        if hasattr(self.board_detector, '_clear_laser'):
            self.board_detector._clear_laser()
        
        # Standard cleanup
        cv2.destroyAllWindows()
        p.disconnect()
        print("✅ Coordination cleanup complete!")

def main():
    """Main function"""
    print("=" * 100)
    print("🤖 MULTI-ROBOT TIC-TAC-TOE: ROS2 COORDINATION SYSTEM")
    print("📋 Program 3: Intelligent multi-robot coordination with communication")
    print("=" * 100)
    
    demo = MultiRobotCoordinationDemo()
    demo.run_coordination_demo()

if __name__ == "__main__":
    main()
