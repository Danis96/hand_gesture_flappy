# Hand Gesture Flappy Bird

A fun twist on the classic Flappy Bird game where you control the bird using hand gestures detected by your webcam!

## Features

- **Hand Gesture Control**: Move your hand up and down to control the bird's height
- **Dual Hand Support**: Works with either hand or both hands simultaneously
- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand detection
- **Classic Gameplay**: Traditional Flappy Bird mechanics with pipes and scoring
- **Smooth Controls**: Bird smoothly follows your hand movements

## Requirements

- Python 3.7 or higher
- Webcam/Camera
- Good lighting for hand detection

## Installation

1. **Clone or download this repository**

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Play

1. **Run the game**:
   ```bash
   python hand_gesture_flappy_bird.py
   ```

2. **Make sure your camera is working** - the game will automatically access your default camera

3. **Position yourself in front of the camera** with good lighting

4. **Control the bird**:
   - Hold your hand(s) in front of the camera
   - Move your hand **UP** to make the bird go **UP**
   - Move your hand **DOWN** to make the bird go **DOWN**
   - If both hands are detected, the bird will follow the average position
   - If no hands are detected, gravity will pull the bird down

5. **Avoid the pipes** and try to get the highest score!

6. **Game Controls**:
   - `SPACE` - Restart game when game over
   - `ESC` - Quit the game

## Tips for Best Experience

- **Lighting**: Ensure you have good, even lighting on your hands
- **Background**: A contrasting background helps with hand detection
- **Distance**: Keep your hands at a comfortable distance from the camera (arm's length)
- **Hand Position**: Keep your hands clearly visible and avoid blocking them with other objects

## Troubleshooting

- **Camera not working**: Make sure no other applications are using your camera
- **Hand not detected**: Check lighting and make sure your hands are clearly visible
- **Laggy performance**: Close other applications to free up system resources
- **Game too sensitive**: You can adjust the smoothing factor in the code (line 71: `self.y += diff * 0.1`)

## Game Mechanics

- The bird is controlled by your hand height position
- Pipes spawn every 2 seconds with random gap positions
- Score increases by 1 for each pipe you successfully pass
- Game ends if you hit a pipe or the screen boundaries
- The bird smoothly follows your hand movements for better control

## Customization

You can easily modify the game by changing the constants at the top of the file:

- `SCREEN_WIDTH` / `SCREEN_HEIGHT`: Change game window size
- `GRAVITY`: Adjust how fast the bird falls when no hand is detected
- `PIPE_SPEED`: Change how fast pipes move
- `PIPE_GAP`: Adjust the gap size between pipes
- Colors and other visual elements

## Dependencies

- **pygame**: Game engine and graphics
- **opencv-python**: Camera access and image processing
- **mediapipe**: Hand tracking and gesture recognition
- **numpy**: Mathematical operations

Enjoy playing Hand Gesture Flappy Bird! 🐦✋ 