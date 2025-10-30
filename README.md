# Intelligent Multi-Robot Coordination System with Real-Time Computer Vision and ROS2

<div align="center">

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04%20LTS-orange)](https://ubuntu.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A sophisticated multi-robot coordination platform demonstrating advanced robotics concepts through intelligent gameplay between two 7-DOF KUKA IIWA robotic arms**

</div>

---

## 📋 Project Overview

This project presents a comprehensive multi-robot coordination system that integrates **computer vision**, **inverse kinematics**, **ROS2 communication protocols**, and **collision avoidance** to orchestrate intelligent gameplay between two KUKA IIWA robotic arms in a PyBullet simulation environment. This system serves as both an educational demonstration platform and a research foundation for collaborative robotics.

### Key Highlights

- **🤖 Dual 7-DOF Robot Coordination**: Synchronized control of two KUKA IIWA robotic arms with complete joint articulation
- **👁️ Multi-Camera Computer Vision**: Real-time board detection using 5-camera fusion
- **🔗 ROS2 Communication**: Message passing with custom protocols for robot coordination
- **🎭 Theatrical Movement System**: Sequential joint demonstrations showcasing all degrees of freedom
- **🛡️ Safety Systems**: Real-time collision detection and emergency stop protocols
- **🎮 Interactive Gameplay**: Turn-based coordination with win/loss/draw detection

---

## 🎯 Features

### Core Robotics Capabilities

- **Advanced Inverse Kinematics**: Damped least-squares IK solver with singularity avoidance
- **Trajectory Planning**: Multi-waypoint path generation with smooth interpolation
- **Collision Avoidance**: Spatial analysis with 0.3m safety threshold and emergency protocols
- **Joint Control**: Precise position control with ±2mm end-effector accuracy
- **Theatrical Demonstrations**: Sequential activation of all 7 joints for educational impact

### Computer Vision System

- **Multi-Camera Fusion**: Bayesian confidence scoring across 5 synchronized camera perspectives
- **Board Detection**: Hough transform-based line detection with grid extraction
- **Symbol Recognition**: Template matching for X/O detection
- **Real-Time Processing**: <100ms latency for complete vision pipeline
- **Adaptive Algorithms**: Automatic adjustment for varying lighting conditions

### Communication & Coordination

- **ROS2 Humble Integration**: Custom message types and service interfaces
- **Turn-Based Logic**: Intelligent game state management with rule enforcement
- **Status Monitoring**: Real-time robot status and performance metrics
- **Emergency Systems**: Sub-50ms emergency stop with coordinated safety protocols
- **Event-Driven Architecture**: Asynchronous message passing for system coordination

---

## 🏗️ System Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                   User Interface Layer                      │
│  (PyBullet 3D Visualization + Multi-Camera OpenCV Windows)  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                 ROS2 Communication Layer                    │
│     (Message Passing + Services + QoS Management)           │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────┬───────────────────┬────────────────────────┐
│ Vision System   │ Robot Control     │ Game Logic Engine      │
│                 │                   │                        │
│ • Multi-Camera  │ • IK Solver       │ • Rule Enforcement     │
│ • Detection     │ • Trajectory      │ • Win/Draw Detection   │
│ • Recognition   │ • Safety Monitor  │ • State Management     │
└─────────────────┴───────────────────┴────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│              PyBullet Physics Simulation                    │
│        (KUKA IIWA Models + Collision Detection)             │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Core Frameworks**
- **PyBullet 3.2.5+**: Physics simulation and robot dynamics
- **ROS2 Humble**: Distributed robotics communication (LTS)
- **OpenCV 4.6+**: Computer vision and image processing
- **NumPy 1.21+**: Mathematical computations and matrix operations

**Development Environment**
- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Python**: 3.10+
- **Graphics**: OpenGL 3.3+ for 3D rendering

---

## 🚀 Installation

### Prerequisites

Ensure your system meets these requirements:
- Ubuntu 22.04 LTS
- Python 3.10 or higher
- At least 8GB RAM (16GB recommended)
- Graphics card with OpenGL 3.3+ support

### Step 1: Clone Repository

```bash
cd ~/Desktop
git clone https://github.com/CodeKunalTomar/multi_robot_coordination_system.git
cd multi_robot_coordination_system
```

### Step 2: Install ROS2 Humble

```bash
# Add ROS2 repository
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble
sudo apt update
sudo apt install ros-humble-desktop -y

# Source ROS2 setup
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install pybullet==3.2.5
pip install opencv-python==4.6.0.66
pip install numpy==1.21.6
pip install scipy==1.8.1
pip install matplotlib==3.5.3

# Install ROS2 Python dependencies
sudo apt install python3-colcon-common-extensions python3-rosdep -y
sudo rosdep init
rosdep update
```

### Step 4: Build ROS2 Workspace

```bash
# Initialize workspace
cd ~/Desktop/multi_robot_tictactoe
colcon build --symlink-install

# Source workspace
echo "source ~/Desktop/multi_robot_tictactoe/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Verify Installation

```bash
# Test PyBullet
python3 -c "import pybullet as p; print('PyBullet version:', p.getVersionInfo())"

# Test OpenCV
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Test ROS2
ros2 --version
```

---

## 💻 Usage

### Quick Start

```bash
# Navigate to project directory
cd ~/Desktop/multi_robot_coordination_system

# Run the complete coordination system (Program 3)
python3 src/communication/ros2_coordinator.py
```

### Program Variants

The system includes multiple demonstration programs showcasing progressive features:

#### Program 1: Basic Environment Setup
```bash
# Dual-arm environment with multi-camera system
python3 src/simulation/dual_arm_environment.py
```

#### Program 2: Computer Vision Integration
```bash
# Board detection with theatrical joint movements
python3 src/vision/board_detector.py
```

#### Program 3: Full Coordination System
```bash
# Complete multi-robot coordination with ROS2
python3 src/communication/ros2_coordinator.py
```

### Interactive Controls

Once the system is running, use these keyboard controls:

**Game Controls**
- `R` - Reset game state
- `X` + `1-9` - Player X makes move (only on X's turn)
- `O` + `1-9` - Player O makes move (only on O's turn)
- `C` - Clear all virtual symbols
- `V` - Cycle through camera viewpoints (5 perspectives)
- `1-5` - Focus on specific camera feed
- `D` - Toggle detection overlay display
- `G` - Print current game state to console

---

## 🔮 Future Development

### Version 3.0 Roadmap

#### AI Integration
- **Minimax Algorithm**: Strategic gameplay with alpha-beta pruning
- **Machine Learning**: Neural network-based move optimization
- **Adaptive Difficulty**: Multiple AI opponent levels for educational scenarios
- **Strategy Analysis**: Move evaluation and game tree visualization

#### Physical Robot Integration
- **Hardware Interface**: Real KUKA IIWA controller integration
- **Safety Systems**: Physical collision detection and force limiting
- **Computer Vision**: Real camera hardware and lighting management
- **Calibration**: Automated camera-robot calibration procedures

#### Extended Capabilities
- **Multi-Game Support**: Chess, Connect Four, and other board games
- **Tournament System**: Multi-player competition framework
- **Cloud Integration**: Remote demonstration and monitoring capabilities
- **Mobile Control**: Tablet/smartphone interface for system control

### Research Extensions

#### Advanced Computer Vision
- Deep learning-based board detection
- Real-time hand gesture recognition for human interaction
- 3D reconstruction for complex object manipulation
- Semantic understanding of game states

#### Enhanced Coordination
- Three or more robot collaboration
- Swarm intelligence algorithms for multi-agent systems
- Distributed decision-making frameworks
- Human-robot-robot interaction paradigms

#### Performance Optimization
- GPU acceleration for vision processing
- Real-time trajectory optimization
- Predictive collision avoidance
- Energy-efficient motion planning

---

## 🏆 Project Achievements

### Technical Milestones
- ✅ Complete 7-DOF multi-robot coordination system
- ✅ Real-time computer vision
- ✅ ROS2 Humble integration with custom protocols
- ✅ Sub-second response times with collision avoidance

### Technical Innovation
- Novel theatrical joint movement system
- Multi-camera Bayesian fusion algorithm
- Educational robotics demonstration platform
- Industry-standard ROS2 architecture

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Use**: This software is developed for educational and research purposes. Commercial use requires explicit permission.

---

## 🙏 Acknowledgments

### Technical Resources
- **KUKA Robotics**: KUKA IIWA robot models and documentation
- **ROS2 Community**: Humble Hawksbill distribution and support
- **PyBullet Team**: Open-source physics simulation framework
- **OpenCV Foundation**: Computer vision library and algorithms

### Inspiration
- "The Duel: Timo Boll vs. KUKA Robot" - Demonstrating precision robotics capabilities
- ROS-Industrial Initiative - Industrial robotics standards
- Academic Robotics Research Community

---

## 📞 Support & Contribution

### Reporting Issues
For bugs, feature requests, or questions:
1. Check [existing issues](https://github.com/CodeKunalTomar/multi_robot_coordination_system/issues)
2. Create a new issue with detailed description
3. Include system information and error logs

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes with clear messages
4. Submit a pull request with description

### Contact for Collaboration
Interested in extending this work? Contact me for:
- Research collaborations
- Feature additions
- Academic consultations
- Industry partnerships

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ for robotics education and research

</div>
