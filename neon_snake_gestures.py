import pygame
import cv2
import mediapipe as mp
import numpy as np
import random
import math
from typing import Tuple, List, Optional, Dict
from enum import Enum
import time

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH: int = 1400
SCREEN_HEIGHT: int = 800
GAME_WIDTH: int = 900
GAME_HEIGHT: int = 800
CAMERA_WIDTH: int = 500
CAMERA_HEIGHT: int = 400
FPS: int = 60

# Grid settings
GRID_SIZE: int = 20
GRID_WIDTH: int = GAME_WIDTH // GRID_SIZE
GRID_HEIGHT: int = GAME_HEIGHT // GRID_SIZE

# Colors - Neon Theme
BLACK: Tuple[int, int, int] = (0, 0, 0)
DARK_BLUE: Tuple[int, int, int] = (10, 10, 40)
NEON_GREEN: Tuple[int, int, int] = (57, 255, 20)
NEON_BLUE: Tuple[int, int, int] = (0, 255, 255)
NEON_PINK: Tuple[int, int, int] = (255, 20, 147)
NEON_PURPLE: Tuple[int, int, int] = (191, 64, 191)
NEON_YELLOW: Tuple[int, int, int] = (255, 255, 0)
NEON_ORANGE: Tuple[int, int, int] = (255, 165, 0)
WHITE: Tuple[int, int, int] = (255, 255, 255)
GLOW_GREEN: Tuple[int, int, int] = (57, 255, 20, 100)
GLOW_PINK: Tuple[int, int, int] = (255, 20, 147, 100)

# Game settings
INITIAL_SPEED: float = 150  # milliseconds between moves
MIN_SPEED: float = 50       # fastest speed
SPEED_DECREASE: float = 2   # speed increase per food eaten

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Single hand for direction control
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def get_hand_direction(self, frame: np.ndarray) -> Tuple[Optional[Direction], np.ndarray]:
        """Detect pinch gesture and direction for precise control"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        debug_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        direction: Optional[Direction] = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(debug_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get key landmarks (thumb tip and index finger tip)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * frame.shape[1]
                    y = landmark.y * frame.shape[0]
                    landmarks.append((x, y))
                
                if len(landmarks) >= 21:  # Make sure we have all landmarks
                    # Get thumb tip (landmark 4) and index finger tip (landmark 8)
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    
                    # Calculate distance between thumb and index finger
                    distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
                    
                    # Pinch threshold (adjust based on camera resolution)
                    pinch_threshold = 30
                    
                    if distance < pinch_threshold:
                        # Pinch detected! Now get the pinch point direction
                        pinch_x = (thumb_tip[0] + index_tip[0]) / 2
                        pinch_y = (thumb_tip[1] + index_tip[1]) / 2
                        
                        # Frame center for reference
                        frame_center_x = frame.shape[1] / 2
                        frame_center_y = frame.shape[0] / 2
                        
                        # Calculate direction from center
                        dx = pinch_x - frame_center_x
                        dy = pinch_y - frame_center_y
                        
                        # Deadzone around center
                        deadzone = 60
                        
                        if abs(dx) > deadzone or abs(dy) > deadzone:
                            # Determine primary direction
                            if abs(dx) > abs(dy):
                                if dx > 0:
                                    direction = Direction.RIGHT
                                    cv2.putText(debug_frame, "PINCH RIGHT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                else:
                                    direction = Direction.LEFT
                                    cv2.putText(debug_frame, "PINCH LEFT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            else:
                                if dy > 0:
                                    direction = Direction.DOWN
                                    cv2.putText(debug_frame, "PINCH DOWN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                else:
                                    direction = Direction.UP
                                    cv2.putText(debug_frame, "PINCH UP", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
                        # Draw pinch point and controls
                        cv2.circle(debug_frame, (int(pinch_x), int(pinch_y)), 20, (0, 255, 255), -1)  # Yellow pinch point
                        cv2.circle(debug_frame, (int(frame_center_x), int(frame_center_y)), deadzone, (255, 255, 255), 2)  # Deadzone
                        
                        # Draw line from center to pinch point
                        cv2.line(debug_frame, (int(frame_center_x), int(frame_center_y)), (int(pinch_x), int(pinch_y)), (0, 255, 255), 3)
                        
                        # Pinch status
                        cv2.putText(debug_frame, "PINCHED!", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(debug_frame, f"Distance: {int(distance)}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                    else:
                        # Not pinched - show instructions
                        cv2.putText(debug_frame, "PINCH to control", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                        cv2.putText(debug_frame, f"Distance: {int(distance)}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Draw thumb and index finger
                        cv2.circle(debug_frame, (int(thumb_tip[0]), int(thumb_tip[1])), 8, (255, 0, 0), -1)  # Red thumb
                        cv2.circle(debug_frame, (int(index_tip[0]), int(index_tip[1])), 8, (0, 0, 255), -1)  # Blue index
                        cv2.line(debug_frame, (int(thumb_tip[0]), int(thumb_tip[1])), (int(index_tip[0]), int(index_tip[1])), (255, 255, 255), 1)
                    
                    # Draw directional guides
                    cv2.putText(debug_frame, "UP", (frame.shape[1] // 2 - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
                    cv2.putText(debug_frame, "DOWN", (frame.shape[1] // 2 - 25, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
                    cv2.putText(debug_frame, "LEFT", (10, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
                    cv2.putText(debug_frame, "RIGHT", (frame.shape[1] - 80, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        return direction, debug_frame

class Food:
    def __init__(self, snake_body: List[Tuple[int, int]]):
        self.position: Tuple[int, int] = self.generate_position(snake_body)
        self.animation: float = 0
        
    def generate_position(self, snake_body: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Generate food position that doesn't overlap with snake"""
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in snake_body:
                return (x, y)
    
    def update(self) -> None:
        """Update food animation"""
        self.animation += 0.2
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw neon glowing food"""
        x = self.position[0] * GRID_SIZE + GRID_SIZE // 2
        y = self.position[1] * GRID_SIZE + GRID_SIZE // 2
        
        # Pulsing glow effect
        pulse = abs(math.sin(self.animation))
        base_radius = GRID_SIZE // 3
        glow_radius = int(base_radius + pulse * 8)
        
        # Draw multiple glow layers
        for i in range(5):
            alpha = int(50 - i * 10)
            if alpha > 0:
                radius = glow_radius + i * 3
                glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*NEON_PINK, alpha), (radius, radius), radius)
                screen.blit(glow_surface, (x - radius, y - radius))
        
        # Draw core food
        pygame.draw.circle(screen, NEON_PINK, (x, y), base_radius)
        pygame.draw.circle(screen, WHITE, (x, y), base_radius // 2)

class Snake:
    def __init__(self):
        self.body: List[Tuple[int, int]] = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction: Direction = Direction.RIGHT
        self.pending_direction: Optional[Direction] = None
        self.grow_pending: bool = False
        self.trail_positions: List[Tuple[Tuple[int, int], float]] = []
        self.animation: float = 0
        
    def change_direction(self, new_direction: Direction) -> None:
        """Change snake direction (prevent 180-degree turns)"""
        if len(self.body) > 1:
            opposite_directions = {
                Direction.UP: Direction.DOWN,
                Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT,
                Direction.RIGHT: Direction.LEFT
            }
            if new_direction != opposite_directions.get(self.direction):
                self.pending_direction = new_direction
        else:
            self.pending_direction = new_direction
    
    def move(self) -> bool:
        """Move snake and return True if collision occurred"""
        # Apply pending direction change
        if self.pending_direction:
            self.direction = self.pending_direction
            self.pending_direction = None
        
        # Calculate new head position
        head_x, head_y = self.body[0]
        
        if self.direction == Direction.UP:
            head_y -= 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        elif self.direction == Direction.RIGHT:
            head_x += 1
        
        new_head = (head_x, head_y)
        
        # Check wall collision
        if (head_x < 0 or head_x >= GRID_WIDTH or 
            head_y < 0 or head_y >= GRID_HEIGHT):
            return True
        
        # Check self collision
        if new_head in self.body:
            return True
        
        # Add trail effect
        if len(self.body) > 0:
            self.trail_positions.append((self.body[0], time.time()))
        
        # Remove old trail positions
        current_time = time.time()
        self.trail_positions = [(pos, t) for pos, t in self.trail_positions 
                               if current_time - t < 0.3]
        
        # Move snake
        self.body.insert(0, new_head)
        
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False
        
        return False
    
    def grow(self) -> None:
        """Make snake grow on next move"""
        self.grow_pending = True
    
    def update(self) -> None:
        """Update snake animation"""
        self.animation += 0.3
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw neon glowing snake"""
        # Draw trail
        current_time = time.time()
        for (x, y), timestamp in self.trail_positions:
            age = current_time - timestamp
            alpha = int(255 * (1 - age / 0.3))
            if alpha > 0:
                trail_surface = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                color = (*NEON_GREEN, alpha // 3)
                pygame.draw.rect(trail_surface, color, (0, 0, GRID_SIZE, GRID_SIZE))
                screen.blit(trail_surface, (x * GRID_SIZE, y * GRID_SIZE))
        
        # Draw snake body with glow
        for i, (x, y) in enumerate(self.body):
            screen_x = x * GRID_SIZE
            screen_y = y * GRID_SIZE
            
            # Glow effect
            glow_size = GRID_SIZE + 6
            for j in range(3):
                alpha = 60 - j * 20
                if alpha > 0:
                    glow_surface = pygame.Surface((glow_size + j * 4, glow_size + j * 4), pygame.SRCALPHA)
                    color = (*NEON_GREEN, alpha)
                    pygame.draw.rect(glow_surface, color, (0, 0, glow_size + j * 4, glow_size + j * 4), border_radius=5)
                    screen.blit(glow_surface, (screen_x - j * 2 - 3, screen_y - j * 2 - 3))
            
            # Snake segment color based on position
            if i == 0:  # Head
                color = NEON_YELLOW
                # Add pulsing effect to head
                pulse = abs(math.sin(self.animation))
                brightness = int(200 + pulse * 55)
                color = (min(255, brightness), min(255, brightness), 0)
            else:  # Body
                # Gradient from head to tail
                intensity = max(100, 255 - i * 10)
                color = (0, intensity, 0)
            
            # Draw segment
            pygame.draw.rect(screen, color, (screen_x + 2, screen_y + 2, GRID_SIZE - 4, GRID_SIZE - 4), border_radius=3)
            
            # Add inner highlight
            if i == 0:
                pygame.draw.rect(screen, WHITE, (screen_x + 6, screen_y + 6, GRID_SIZE - 12, GRID_SIZE - 12), border_radius=2)

class NeonSnakeGame:
    def __init__(self):
        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🐍 NEON SNAKE - Hand Gesture Control 🐍")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        # Game objects
        self.snake: Snake = Snake()
        self.food: Food = Food(self.snake.body)
        
        # Game state
        self.score: int = 0
        self.high_score: int = 0
        self.game_over: bool = False
        self.last_move_time: float = time.time()
        self.move_delay: float = INITIAL_SPEED / 1000.0  # Convert to seconds
        
        # Hand tracking
        self.hand_tracker: HandTracker = HandTracker()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Fonts
        self.font: pygame.font.Font = pygame.font.Font(None, 36)
        self.big_font: pygame.font.Font = pygame.font.Font(None, 72)
        self.small_font: pygame.font.Font = pygame.font.Font(None, 24)
        
        # Visual effects
        self.background_animation: float = 0
        
    def get_hand_input(self) -> Optional[Direction]:
        """Get hand gesture input"""
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)
        direction, debug_frame = self.hand_tracker.get_hand_direction(frame)
        
        # Resize camera feed for display
        debug_frame_resized = cv2.resize(debug_frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        
        # Convert to pygame surface and display
        debug_frame_rgb = cv2.cvtColor(debug_frame_resized, cv2.COLOR_BGR2RGB)
        debug_surface = pygame.surfarray.make_surface(debug_frame_rgb.swapaxes(0, 1))
        self.screen.blit(debug_surface, (GAME_WIDTH, 0))
        
        return direction
    
    def update_game_speed(self) -> None:
        """Update game speed based on score (progressive difficulty)"""
        new_delay = max(MIN_SPEED, INITIAL_SPEED - self.score * SPEED_DECREASE) / 1000.0
        self.move_delay = new_delay
    
    def check_food_collision(self) -> bool:
        """Check if snake ate food"""
        if self.snake.body[0] == self.food.position:
            self.snake.grow()
            self.score += 1
            self.food = Food(self.snake.body)
            self.update_game_speed()
            return True
        return False
    
    def draw_neon_background(self) -> None:
        """Draw animated neon background"""
        self.background_animation += 0.02
        
        # Dark background with slight gradient
        for y in range(GAME_HEIGHT):
            intensity = int(10 + 5 * math.sin(y * 0.01 + self.background_animation))
            color = (intensity // 4, intensity // 4, intensity)
            pygame.draw.line(self.screen, color, (0, y), (GAME_WIDTH, y))
        
        # Draw grid lines with glow
        grid_alpha = int(30 + 10 * math.sin(self.background_animation))
        
        # Vertical lines
        for x in range(0, GAME_WIDTH, GRID_SIZE):
            if x % (GRID_SIZE * 5) == 0:  # Every 5th line brighter
                color = (*NEON_BLUE, min(60, grid_alpha + 20))
            else:
                color = (*DARK_BLUE, grid_alpha)
            
            line_surface = pygame.Surface((2, GAME_HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(line_surface, color, (0, 0), (0, GAME_HEIGHT))
            self.screen.blit(line_surface, (x, 0))
        
        # Horizontal lines
        for y in range(0, GAME_HEIGHT, GRID_SIZE):
            if y % (GRID_SIZE * 5) == 0:  # Every 5th line brighter
                color = (*NEON_BLUE, min(60, grid_alpha + 20))
            else:
                color = (*DARK_BLUE, grid_alpha)
            
            line_surface = pygame.Surface((GAME_WIDTH, 2), pygame.SRCALPHA)
            pygame.draw.line(line_surface, color, (0, 0), (GAME_WIDTH, 0))
            self.screen.blit(line_surface, (0, y))
    
    def draw_ui(self) -> None:
        """Draw game UI"""
        # Game area border
        border_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
        pygame.draw.rect(self.screen, NEON_PURPLE, border_rect, 3)
        
        # Camera feed border
        camera_rect = pygame.Rect(GAME_WIDTH, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
        pygame.draw.rect(self.screen, NEON_BLUE, camera_rect, 3)
        
        # Score display
        score_text = self.font.render(f"SCORE: {self.score}", True, NEON_GREEN)
        high_score_text = self.font.render(f"HIGH: {self.high_score}", True, NEON_YELLOW)
        speed_text = self.small_font.render(f"Speed: {int((INITIAL_SPEED - self.move_delay * 1000) / SPEED_DECREASE + 1)}", True, NEON_PURPLE)
        
        # Add glow to score text
        for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
            glow_text = self.font.render(f"SCORE: {self.score}", True, (0, 100, 0))
            self.screen.blit(glow_text, (GAME_WIDTH + 20 + offset[0], CAMERA_HEIGHT + 20 + offset[1]))
        
        self.screen.blit(score_text, (GAME_WIDTH + 20, CAMERA_HEIGHT + 20))
        self.screen.blit(high_score_text, (GAME_WIDTH + 20, CAMERA_HEIGHT + 60))
        self.screen.blit(speed_text, (GAME_WIDTH + 20, CAMERA_HEIGHT + 100))
        
                 # Controls instructions
        control_texts = [
            "🐍 NEON SNAKE CONTROLS:",
            "",
            "🤏 PINCH CONTROL:",
            "• PINCH thumb + index finger",
            "• MOVE pinched fingers to steer", 
            "• Like a tiny joystick!",
            "",
            "🎯 Precise directional control:",
            "• Pinch and move LEFT = Turn LEFT",
            "• Pinch and move RIGHT = Turn RIGHT",
            "• Pinch and move UP = Turn UP", 
            "• Pinch and move DOWN = Turn DOWN",
            "",
            "📱 Keep hand visible in camera",
            "🔴 Red thumb, 🔵 Blue index finger",
            "",
            "SPACE: Restart | ESC: Quit"
        ]
        
        for i, text in enumerate(control_texts):
            color = NEON_YELLOW if i == 0 else NEON_BLUE if text.startswith("•") else WHITE
            if text:  # Skip empty lines
                control_surface = self.small_font.render(text, True, color)
                self.screen.blit(control_surface, (GAME_WIDTH + 20, CAMERA_HEIGHT + 140 + i * 25))
        
        # Length display
        length_text = self.font.render(f"LENGTH: {len(self.snake.body)}", True, NEON_ORANGE)
        self.screen.blit(length_text, (GAME_WIDTH + 20, CAMERA_HEIGHT + 500))
        
        if self.game_over:
            # Game over overlay
            overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            # Game over text with glow
            game_over_text = self.big_font.render("GAME OVER", True, NEON_PINK)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, NEON_GREEN)
            restart_text = self.font.render("Press SPACE to restart", True, WHITE)
            
            # Add glow effect
            for offset in [(3, 3), (-3, -3), (3, -3), (-3, 3)]:
                glow_game_over = self.big_font.render("GAME OVER", True, (100, 0, 50))
                self.screen.blit(glow_game_over, (GAME_WIDTH // 2 - 150 + offset[0], GAME_HEIGHT // 2 - 100 + offset[1]))
            
            self.screen.blit(game_over_text, (GAME_WIDTH // 2 - 150, GAME_HEIGHT // 2 - 100))
            self.screen.blit(final_score_text, (GAME_WIDTH // 2 - 100, GAME_HEIGHT // 2 - 30))
            self.screen.blit(restart_text, (GAME_WIDTH // 2 - 120, GAME_HEIGHT // 2 + 20))
    
    def restart_game(self) -> None:
        """Restart the game"""
        if self.score > self.high_score:
            self.high_score = self.score
        
        self.snake = Snake()
        self.food = Food(self.snake.body)
        self.score = 0
        self.game_over = False
        self.last_move_time = time.time()
        self.move_delay = INITIAL_SPEED / 1000.0
    
    def run(self) -> None:
        """Main game loop"""
        running: bool = True
        
        print("🐍 NEON SNAKE - Pinch Control Edition! 🐍")
        print("📋 Instructions:")
        print("🤏 Use PINCH gesture to control snake direction:")
        print("• PINCH thumb and index finger together")
        print("• MOVE the pinched fingers like a tiny joystick")
        print("• Move pinched fingers LEFT = Snake turns LEFT")
        print("• Move pinched fingers RIGHT = Snake turns RIGHT")
        print("• Move pinched fingers UP = Snake turns UP")
        print("• Move pinched fingers DOWN = Snake turns DOWN")
        print("\n🎯 Visual feedback:")
        print("🔴 Red dot = Thumb tip")
        print("🔵 Blue dot = Index finger tip")
        print("🟡 Yellow dot = Pinch point (when pinched)")
        print("⚪ White circle = Deadzone")
        print("\n🍎 Eat the pink glowing food to grow")
        print("🚀 Speed increases as you get longer!")
        print("\n⌨️ Controls:")
        print("- SPACE: Restart when game over")
        print("- ESC: Quit")
        
        while running:
            current_time = time.time()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and self.game_over:
                        self.restart_game()
            
            if not self.game_over:
                # Get hand input
                hand_direction = self.get_hand_input()
                if hand_direction:
                    self.snake.change_direction(hand_direction)
                
                # Move snake based on timing
                if current_time - self.last_move_time >= self.move_delay:
                    collision = self.snake.move()
                    if collision:
                        self.game_over = True
                    else:
                        self.check_food_collision()
                    self.last_move_time = current_time
                
                # Update animations
                self.snake.update()
                self.food.update()
            else:
                # Still show camera when game over
                self.get_hand_input()
            
            # Draw everything
            self.draw_neon_background()
            
            if not self.game_over:
                self.snake.draw(self.screen)
                self.food.draw(self.screen)
            
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = NeonSnakeGame()
    game.run() 