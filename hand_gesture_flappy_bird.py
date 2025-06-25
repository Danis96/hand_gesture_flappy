import pygame
import cv2
import mediapipe as mp
import numpy as np
import random
import math
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
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def get_hand_positions(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Get hand positions from camera image
        Returns list of (x, y) coordinates for detected hands
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        hand_positions: List[Tuple[float, float]] = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the middle finger MCP joint (landmark 9) as reference point
                landmark = hand_landmarks.landmark[9]
                x: float = landmark.x
                y: float = landmark.y
                hand_positions.append((x, y))
                
        return hand_positions

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
            self.y += diff * 0.1
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
        pygame.display.set_caption("Hand Gesture Flappy Bird")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        # Game objects
        self.bird: Bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes: List[Pipe] = []
        self.score: int = 0
        self.game_over: bool = False
        
        # Hand tracking
        self.hand_tracker: HandTracker = HandTracker()
        self.camera = cv2.VideoCapture(0)
        
        # Fonts
        self.font: pygame.font.Font = pygame.font.Font(None, 36)
        self.big_font: pygame.font.Font = pygame.font.Font(None, 72)
        
        # Spawn first pipe
        self.pipe_timer: int = 0
        
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
        hand_positions = self.hand_tracker.get_hand_positions(frame)
        
        if hand_positions:
            # Use average of all detected hands
            avg_y = sum(pos[1] for pos in hand_positions) / len(hand_positions)
            return avg_y
        
        return None
        
    def draw_hud(self) -> None:
        """Draw game HUD"""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Instructions
        instruction_text = self.font.render("Move your hand up/down to control the bird!", True, WHITE)
        self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 30))
        
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
        
    def run(self) -> None:
        """Main game loop"""
        running: bool = True
        
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
                        
            if not self.game_over:
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
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run() 