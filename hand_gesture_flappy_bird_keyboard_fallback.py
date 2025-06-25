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

class Bird:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.velocity: float = 0
        self.size: int = BIRD_SIZE
        self.target_y: int = y
        
    def update(self, keyboard_input: Optional[str] = None) -> None:
        """Update bird position based on keyboard input or gravity"""
        if keyboard_input == "up":
            # Keyboard control - move up
            self.velocity = -8  # Strong upward movement
        elif keyboard_input == "down":
            # Keyboard control - move down
            self.velocity = 8   # Strong downward movement
        else:
            # Apply gravity when no input detected
            self.velocity += GRAVITY
            self.y += int(self.velocity)
            
        # Apply velocity
        if keyboard_input:
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
        pygame.display.set_caption("Keyboard Flappy Bird - Test Version")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        # Game objects
        self.bird: Bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes: List[Pipe] = []
        self.score: int = 0
        self.game_over: bool = False
        
        # Fonts
        self.font: pygame.font.Font = pygame.font.Font(None, 36)
        self.big_font: pygame.font.Font = pygame.font.Font(None, 72)
        
        # Game state
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
        
    def get_keyboard_input(self, keys) -> Optional[str]:
        """Get keyboard input for bird control"""
        if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]:
            return "up"
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            return "down"
        return None
        
    def draw_hud(self) -> None:
        """Draw game HUD"""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Instructions
        instruction_text = self.font.render("Use UP/DOWN arrows, W/S, or SPACE to control the bird!", True, WHITE)
        self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 60))
        
        # Controls
        controls_text = self.font.render("SPACE/R: restart | ESC: quit", True, WHITE)
        self.screen.blit(controls_text, (10, SCREEN_HEIGHT - 30))
        
        if self.game_over:
            # Game over screen
            game_over_text = self.big_font.render("GAME OVER", True, RED)
            restart_text = self.font.render("Press SPACE or R to restart, ESC to quit", True, WHITE)
            
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
        
        print("🎮 Keyboard Flappy Bird - Test Version")
        print("📋 Controls:")
        print("- UP arrow or W or SPACE: Make bird go up")
        print("- DOWN arrow or S: Make bird go down")
        print("- R or SPACE: Restart when game over")
        print("- ESC: Quit")
        
        while running:
            # Get current key states
            keys = pygame.key.get_pressed()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif (event.key == pygame.K_SPACE or event.key == pygame.K_r) and self.game_over:
                        self.restart_game()
                        
            if not self.game_over:
                # Get keyboard input
                keyboard_input = self.get_keyboard_input(keys)
                
                # Update bird
                self.bird.update(keyboard_input)
                
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
            
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run() 