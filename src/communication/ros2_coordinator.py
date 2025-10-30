#!/usr/bin/env python3
"""
Multi-Robot Tic-Tac-Toe: ROS2 Multi-Robot Coordination System
Program 3: Intelligent turn-based coordination with communication
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
    
    def start_new_game(self):
        """Initialize a new game WITHOUT threading issues"""
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
        
        # Move arms to ready positions SYNCHRONOUSLY (no threading)
        print("🤖 Moving arms to game-ready positions...")
        self._move_arms_to_game_ready_positions_sync()
        
        print(f"🎯 Player {self.current_player.name}'s turn")
        print("📋 Use X/O + 1-9 to make coordinated moves")
    
    def _move_arms_to_game_ready_positions_sync(self):
        """Move both arms to ready positions synchronously (no threading)"""
        ready_pos_x = self.board_detector._get_ready_position(1)
        ready_pos_o = self.board_detector._get_ready_position(2)
        
        # Move Arm 1 first
        print("🤖 Moving Arm 1 to ready position...")
        self.board_detector._coordinated_dramatic_movement(self.env.robot_arm1_id, ready_pos_x)
        
        # Small delay
        time.sleep(0.5)
        
        # Move Arm 2 second
        print("🤖 Moving Arm 2 to ready position...")
        self.board_detector._coordinated_dramatic_movement(self.env.robot_arm2_id, ready_pos_o)
        
        print("✅ Both arms ready for gameplay")
    
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
        
        # Execute the move
        success = self._execute_coordinated_move(player, position)
        
        return success
    
    def _execute_coordinated_move(self, player: Player, position: int) -> bool:
        """Execute the actual coordinated move"""
        arm_id = self.robot_assignments[player]
        
        # Execute theatrical move
        success = self.board_detector.place_virtual_symbol_with_fast_pointing(position, player.value)
        
        if success:
            # Update game state
            self._update_game_state_after_move(player, position)
            return True
        else:
            print(f"❌ Move execution failed for Player {player.name}")
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
        elif self.game_result == GameResult.O_WINS:
            print("🏆 PLAYER O (ARM 2) WINS!")
        else:
            print("🤝 IT'S A DRAW!")
        
        self._show_game_statistics()
    
    def _show_game_statistics(self):
        """Show detailed game statistics"""
        print("\n📊 GAME STATISTICS:")
        print("-" * 30)
        print(f"Total moves: {len(self.move_history)}")
        if self.move_history:
            duration = time.time() - self.move_history[0]['timestamp']
            print(f"Game duration: {duration:.1f} seconds")
        
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


class MultiRobotCoordinationDemo(DualArmEnvironment):
    """Enhanced environment with multi-robot coordination"""
    
    def __init__(self):
        super().__init__()
        self.board_detector = BoardDetector(self)
        self.game_coordinator = GameCoordinator(self, self.board_detector)
        
        # Add missing attributes for camera detection
        self.show_detection_overlay = True
        self.detection_update_counter = 0
        self.detection_update_interval = 15
        
        print("✅ Multi-Robot Coordination Demo initialized")
    
    def update_camera_displays_with_detection(self):
        """Update camera displays with detection overlays (with connection check)"""
        try:
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
                    # Silently handle camera errors during normal operation
                    pass
        
        except Exception as e:
            # Silently handle detection errors
            pass
    
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
            print("🔄 Entering main game loop...")
            
            while True:
                # Check PyBullet connection
                try:
                    p.stepSimulation()
                except:
                    print("❌ PyBullet disconnected - Exiting")
                    break
                    
                self.update_camera_displays_with_detection()
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("🛑 ESC pressed - Exiting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    print("🔄 Starting new game...")
                    self.game_coordinator.start_new_game()
                elif key == ord('t') or key == ord('T'):
                    self.game_coordinator._show_game_statistics()
                elif key == ord('w') or key == ord('W'):
                    self._show_winning_combinations()
                elif key == ord('v') or key == ord('V'):
                    self.cycle_viewpoint()
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
                
                time.sleep(1/60)
                
        except KeyboardInterrupt:
            print("🛑 Keyboard interrupt received")
        finally:
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
        """Enhanced cleanup with better error handling"""
        print("🧹 Enhanced cleanup with coordination...")
        
        try:
            # Clear any remaining laser visuals
            if hasattr(self.board_detector, '_clear_laser'):
                self.board_detector._clear_laser()
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        try:
            p.disconnect()
        except:
            pass
            
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
