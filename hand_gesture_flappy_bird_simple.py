import pygame
import cv2
import numpy as np
import random
from typing import Tuple, List, Optional

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH: int = 800
SCREEN_HEIGHT: int = 600
FPS: int = 60

# Colors
WHITE: Tuple[int, int, int] = (255, 255, 255)
BLACK: Tuple[int, int, int] = (0, 0, 0)
GREEN: Tuple[int, int, int] = (0, 255, 0)
RED: Tuple[int, int, int] = (255, 0, 0)
BLUE: Tuple[int, int, int] = (0, 0, 255)
YELLOW: Tuple[int, int, int] = (255, 255, 0)
PIPE_COLOR: Tuple[int, int, int] = (34, 139, 34)

# Game settings
GRAVITY: float = 0.5
BIRD_SIZE: int = 30
PIPE_WIDTH: int = 80
PIPE_GAP: int = 200
PIPE_SPEED: int = 3

class HandTracker:
    def __init__(self):
        self.calibrated: bool = False
        self.skin_lower: np.ndarray = np.array([0, 20, 70])
        self.skin_upper: np.ndarray = np.array([20, 255, 255])
        self.calibration_frames: int = 0
        self.calibration_samples: List[np.ndarray] = []
        
    def calibrate_skin_color(self, frame: np.ndarray, center_x: int, center_y: int) -> None:
        """Calibrate skin color from a center region"""
        # Sample a small region in the center of the frame
        sample_size: int = 30
        x1: int = max(0, center_x - sample_size)
        x2: int = min(frame.shape[1], center_x + sample_size)
        y1: int = max(0, center_y - sample_size)
        y2: int = min(frame.shape[0], center_y + sample_size)
        
        sample_region = frame[y1:y2, x1:x2]
        hsv_sample = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV values
        mean_hsv = np.mean(hsv_sample.reshape(-1, 3), axis=0)
        self.calibration_samples.append(mean_hsv)
        
        self.calibration_frames += 1
        
        if self.calibration_frames >= 10:  # Calibrate over 10 frames
            # Calculate average HSV values
            avg_hsv = np.mean(self.calibration_samples, axis=0)
            
            # Set HSV range based on calibrated values
            h_range: int = 10
            s_range: int = 60
            v_range: int = 60
            
            self.skin_lower = np.array([
                max(0, avg_hsv[0] - h_range),
                max(20, avg_hsv[1] - s_range),
                max(50, avg_hsv[2] - v_range)
            ])
            
            self.skin_upper = np.array([
                min(179, avg_hsv[0] + h_range),
                255,
                255
            ])
            
            self.calibrated = True
            print(f"Calibration complete! HSV range: {self.skin_lower} - {self.skin_upper}")
    
    def get_hand_position(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Get hand position using color tracking"""
        if not self.calibrated:
            return None
            
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assumed to be the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only consider contours above a minimum size
            if cv2.contourArea(largest_contour) > 1000:
                # Get the center of mass
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Normalize coordinates
                    x_norm = cx / frame.shape[1]
                    y_norm = cy / frame.shape[0]
                    
                    return (x_norm, y_norm)
        
        return None

class Bird:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.velocity: float = 0
        self.size: int = BIRD_SIZE
        self.target_y: int = y
        
    def update(self, hand_y: Optional[float] = None) -> None:
        """Update bird position based on hand input or gravity"""
        if hand_y is not None:
            # Convert hand y (0-1) to screen coordinates (inverted)
            self.target_y = int((1 - hand_y) * SCREEN_HEIGHT)
            # Smooth movement towards target
            diff: float = self.target_y - self.y
            self.y += int(diff * 0.15)  # Slightly faster response
            self.velocity = 0  # Reset velocity when hand is detected
        else:
            # Apply gravity when no hand detected
            self.velocity += GRAVITY
            self.y += int(self.velocity)
            
        # Keep bird on screen
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))
        
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the bird"""
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.size)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.size, 3)
        # Draw eye
        eye_x: int = self.x + 10
        eye_y: int = self.y - 5
        pygame.draw.circle(screen, BLACK, (eye_x, eye_y), 5)
        
    def get_rect(self) -> pygame.Rect:
        """Get bird's collision rectangle"""
        return pygame.Rect(self.x - self.size, self.y - self.size, 
                          self.size * 2, self.size * 2)

class Pipe:
    def __init__(self, x: int):
        self.x: int = x
        self.gap_y: int = random.randint(PIPE_GAP, SCREEN_HEIGHT - PIPE_GAP)
        self.width: int = PIPE_WIDTH
        self.passed: bool = False
        
    def update(self) -> None:
        """Move pipe to the left"""
        self.x -= PIPE_SPEED
        
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the pipe"""
        # Top pipe
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        pygame.draw.rect(screen, PIPE_COLOR, top_rect)
        pygame.draw.rect(screen, BLACK, top_rect, 3)
        
        # Bottom pipe
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, 
                                 self.width, SCREEN_HEIGHT - (self.gap_y + PIPE_GAP // 2))
        pygame.draw.rect(screen, PIPE_COLOR, bottom_rect)
        pygame.draw.rect(screen, BLACK, bottom_rect, 3)
        
    def collides_with(self, bird: Bird) -> bool:
        """Check collision with bird"""
        bird_rect = bird.get_rect()
        
        # Top pipe collision
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        if bird_rect.colliderect(top_rect):
            return True
            
        # Bottom pipe collision
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, 
                                 self.width, SCREEN_HEIGHT - (self.gap_y + PIPE_GAP // 2))
        if bird_rect.colliderect(bottom_rect):
            return True
            
        return False
        
    def is_off_screen(self) -> bool:
        """Check if pipe is off screen"""
        return self.x + self.width < 0

class Game:
    def __init__(self):
        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hand Gesture Flappy Bird - M2 Compatible")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        # Game objects
        self.bird: Bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes: List[Pipe] = []
        self.score: int = 0
        self.game_over: bool = False
        
        # Hand tracking
        self.hand_tracker: HandTracker = HandTracker()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Fonts
        self.font: pygame.font.Font = pygame.font.Font(None, 36)
        self.big_font: pygame.font.Font = pygame.font.Font(None, 72)
        
        # Game state
        self.pipe_timer: int = 0
        self.calibration_mode: bool = True
        self.show_camera: bool = True
        
    def spawn_pipe(self) -> None:
        """Spawn a new pipe"""
        self.pipes.append(Pipe(SCREEN_WIDTH))
        
    def update_pipes(self) -> None:
        """Update all pipes"""
        for pipe in self.pipes[:]:
            pipe.update()
            
            # Check for scoring
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                
            # Remove off-screen pipes
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
                
            # Check collision
            if pipe.collides_with(self.bird):
                self.game_over = True
                
    def get_hand_control(self) -> Optional[float]:
        """Get hand position for bird control"""
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)  # Mirror image
        
        if self.calibration_mode:
            if not self.hand_tracker.calibrated:
                # Show calibration instructions
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                
                # Draw calibration circle
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 2)
                cv2.putText(frame, "Place hand in circle", (center_x - 100, center_y - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Calibrating... {self.hand_tracker.calibration_frames}/10", 
                           (center_x - 120, center_y + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Auto-calibrate from center region
                self.hand_tracker.calibrate_skin_color(frame, center_x, center_y)
            else:
                self.calibration_mode = False
                print("Calibration complete! Starting game...")
        
        # Show camera feed
        if self.show_camera:
            cv2.imshow('Camera Feed - Press C to toggle', frame)
            cv2.waitKey(1)
        
        # Get hand position after calibration
        if self.hand_tracker.calibrated:
            hand_pos = self.hand_tracker.get_hand_position(frame)
            if hand_pos:
                return hand_pos[1]  # Return y coordinate
        
        return None
        
    def draw_hud(self) -> None:
        """Draw game HUD"""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Instructions
        if self.calibration_mode:
            instruction_text = self.font.render("Calibrating camera... Place hand in green circle", True, WHITE)
        else:
            instruction_text = self.font.render("Move your hand up/down to control the bird!", True, WHITE)
        self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 60))
        
        # Controls
        controls_text = self.font.render("Press C to toggle camera, R to recalibrate", True, WHITE)
        self.screen.blit(controls_text, (10, SCREEN_HEIGHT - 30))
        
        if self.game_over:
            # Game over screen
            game_over_text = self.big_font.render("GAME OVER", True, RED)
            restart_text = self.font.render("Press SPACE to restart or ESC to quit", True, WHITE)
            
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
            
    def restart_game(self) -> None:
        """Restart the game"""
        self.bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.pipe_timer = 0
        
    def recalibrate(self) -> None:
        """Restart calibration"""
        self.hand_tracker = HandTracker()
        self.calibration_mode = True
        print("Recalibrating...")
        
    def run(self) -> None:
        """Main game loop"""
        running: bool = True
        
        print("Starting Hand Gesture Flappy Bird!")
        print("Controls:")
        print("- Move hand up/down to control bird")
        print("- Press C to toggle camera view")
        print("- Press R to recalibrate")
        print("- Press SPACE to restart when game over")
        print("- Press ESC to quit")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and self.game_over:
                        self.restart_game()
                    elif event.key == pygame.K_c:
                        self.show_camera = not self.show_camera
                        if not self.show_camera:
                            cv2.destroyAllWindows()
                    elif event.key == pygame.K_r:
                        self.recalibrate()
                        
            if not self.game_over and not self.calibration_mode:
                # Get hand control
                hand_y = self.get_hand_control()
                
                # Update bird
                self.bird.update(hand_y)
                
                # Spawn pipes
                self.pipe_timer += 1
                if self.pipe_timer > 120:  # Spawn pipe every 2 seconds at 60 FPS
                    self.spawn_pipe()
                    self.pipe_timer = 0
                    
                # Update pipes
                self.update_pipes()
                
                # Check boundaries
                if self.bird.y <= 0 or self.bird.y >= SCREEN_HEIGHT:
                    self.game_over = True
            else:
                # Still get camera feed during calibration
                self.get_hand_control()
                    
            # Draw everything
            self.screen.fill(BLUE)  # Sky background
            
            # Draw bird
            self.bird.draw(self.screen)
            
            # Draw pipes
            for pipe in self.pipes:
                pipe.draw(self.screen)
                
            # Draw HUD
            self.draw_hud()
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run() 