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

class BetterHandTracker:
    def __init__(self):
        self.calibrated: bool = False
        self.hand_positions: List[float] = []
        self.background_frame: Optional[np.ndarray] = None
        self.background_ready: bool = False
        self.frame_count: int = 0
        
        # Motion detection parameters
        self.motion_threshold: int = 25
        self.min_hand_area: int = 1500
        self.max_hand_area: int = 20000
        
    def learn_background(self, frame: np.ndarray) -> bool:
        """Learn the background without hands"""
        if self.frame_count < 30:
            if self.background_frame is None:
                self.background_frame = frame.astype(np.float32)
            else:
                # Running average
                cv2.accumulateWeighted(frame, self.background_frame, 0.1)
            self.frame_count += 1
            return False
        else:
            self.background_ready = True
            self.calibrated = True
            return True
    
    def detect_hand_motion(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], np.ndarray, float]:
        """Detect hand using motion and improved filtering"""
        if not self.background_ready or self.background_frame is None:
            return None, frame, 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_gray = cv2.cvtColor(self.background_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray, background_gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create debug frame
        debug_frame = frame.copy()
        
        # Apply colormap to threshold for visualization
        thresh_colored = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
        debug_frame = cv2.addWeighted(debug_frame, 0.7, thresh_colored, 0.3, 0)
        
        best_hand_pos = None
        confidence = 0.0
        
        if contours:
            # Filter contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_hand_area < area < self.max_hand_area:
                    # Calculate bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filter by aspect ratio (hands are usually not too elongated)
                    if 0.3 < aspect_ratio < 3.0:
                        valid_contours.append((contour, area))
            
            if valid_contours:
                # Sort by area and take the largest
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                best_contour, area = valid_contours[0]
                
                # Calculate confidence based on area
                confidence = min(1.0, area / 8000.0)
                
                # Get the topmost point (likely to be fingers)
                topmost = tuple(best_contour[best_contour[:, :, 1].argmin()][0])
                
                # Get centroid for stability
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Use weighted combination of topmost and centroid
                    final_x = int(0.4 * topmost[0] + 0.6 * cx)
                    final_y = int(0.6 * topmost[1] + 0.4 * cy)
                    
                    # Draw on debug frame
                    cv2.drawContours(debug_frame, [best_contour], -1, (0, 255, 0), 2)
                    cv2.circle(debug_frame, (final_x, final_y), 15, (255, 0, 0), -1)
                    cv2.circle(debug_frame, topmost, 8, (0, 0, 255), -1)
                    cv2.circle(debug_frame, (cx, cy), 5, (255, 255, 0), -1)
                    
                    # Draw bounding rectangle
                    x, y, w, h = cv2.boundingRect(best_contour)
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    
                    # Normalize coordinates
                    x_norm = final_x / frame.shape[1]
                    y_norm = final_y / frame.shape[0]
                    
                    best_hand_pos = (x_norm, y_norm)
        
        # Add info to debug frame
        cv2.putText(debug_frame, f"Confidence: {confidence:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Contours: {len(contours)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if best_hand_pos:
            cv2.putText(debug_frame, f"Hand: ({best_hand_pos[0]:.2f}, {best_hand_pos[1]:.2f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return best_hand_pos, debug_frame, confidence
    
    def get_smoothed_position(self, hand_y: float) -> float:
        """Apply smoothing to hand position"""
        self.hand_positions.append(hand_y)
        
        # Keep only last 5 positions
        if len(self.hand_positions) > 5:
            self.hand_positions.pop(0)
        
        # Return weighted average (more weight to recent positions)
        if len(self.hand_positions) == 1:
            return hand_y
        
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(self.hand_positions):]
        weighted_sum = sum(pos * weight for pos, weight in zip(self.hand_positions, weights))
        return weighted_sum / sum(weights)

class Bird:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.velocity: float = 0
        self.size: int = BIRD_SIZE
        
    def update(self, hand_y: Optional[float] = None, confidence: float = 0.0) -> None:
        """Update bird position"""
        if hand_y is not None and confidence > 0.4:
            # Convert hand y to bird position (inverted)
            target_y = int((1 - hand_y) * SCREEN_HEIGHT)
            
            # Smooth movement towards target
            diff = target_y - self.y
            self.y += int(diff * 0.2)  # Adjust speed here
            
            # Reduce velocity when hand is detected
            self.velocity *= 0.5
        else:
            # Apply gravity
            self.velocity += GRAVITY
            self.y += int(self.velocity)
        
        # Keep bird on screen
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))
        
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the bird"""
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.size)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.size, 3)
        # Eye
        pygame.draw.circle(screen, BLACK, (self.x + 10, self.y - 5), 5)
        
    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle"""
        return pygame.Rect(self.x - self.size, self.y - self.size, 
                          self.size * 2, self.size * 2)

class Pipe:
    def __init__(self, x: int):
        self.x: int = x
        self.gap_y: int = random.randint(PIPE_GAP, SCREEN_HEIGHT - PIPE_GAP)
        self.width: int = PIPE_WIDTH
        self.passed: bool = False
        
    def update(self) -> None:
        self.x -= PIPE_SPEED
        
    def draw(self, screen: pygame.Surface) -> None:
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
        bird_rect = bird.get_rect()
        
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, 
                                 self.width, SCREEN_HEIGHT - (self.gap_y + PIPE_GAP // 2))
        
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)
        
    def is_off_screen(self) -> bool:
        return self.x + self.width < 0

class Game:
    def __init__(self):
        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Better Hand Gesture Flappy Bird")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        self.bird: Bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes: List[Pipe] = []
        self.score: int = 0
        self.game_over: bool = False
        
        self.hand_tracker: BetterHandTracker = BetterHandTracker()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.font: pygame.font.Font = pygame.font.Font(None, 36)
        self.big_font: pygame.font.Font = pygame.font.Font(None, 72)
        
        self.pipe_timer: int = 0
        self.learning_background: bool = True
        self.show_camera: bool = True
        self.current_confidence: float = 0.0
        
    def spawn_pipe(self) -> None:
        self.pipes.append(Pipe(SCREEN_WIDTH))
        
    def update_pipes(self) -> None:
        for pipe in self.pipes[:]:
            pipe.update()
            
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
                
            if pipe.collides_with(self.bird):
                self.game_over = True
                
    def get_hand_control(self) -> Optional[float]:
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)  # Mirror
        
        if self.learning_background:
            if self.hand_tracker.learn_background(frame):
                self.learning_background = False
                print("🎮 Background learned! You can now move your hand to control the bird!")
            
            # Show learning progress
            progress = self.hand_tracker.frame_count / 30.0
            cv2.putText(frame, f"Learning background... {progress:.1%}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Keep your hands OUT of view!", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if self.show_camera:
                cv2.imshow('Hand Tracking - Learning Background', frame)
                cv2.waitKey(1)
            return None
        
        # Detect hand motion
        hand_pos, debug_frame, confidence = self.hand_tracker.detect_hand_motion(frame)
        self.current_confidence = confidence
        
        if self.show_camera:
            cv2.imshow('Hand Tracking - Motion Detection (Press C to toggle)', debug_frame)
            cv2.waitKey(1)
        
        if hand_pos and confidence > 0.3:
            smoothed_y = self.hand_tracker.get_smoothed_position(hand_pos[1])
            return smoothed_y
        
        return None
        
    def draw_hud(self) -> None:
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Detection status
        if self.learning_background:
            status_text = self.font.render("Learning background - keep hands away!", True, YELLOW)
            progress = self.hand_tracker.frame_count / 30.0
            progress_text = self.font.render(f"Progress: {progress:.1%}", True, GREEN)
            self.screen.blit(status_text, (10, 50))
            self.screen.blit(progress_text, (10, 90))
        else:
            confidence_color = GREEN if self.current_confidence > 0.5 else (YELLOW if self.current_confidence > 0.3 else RED)
            confidence_text = self.font.render(f"Hand Detection: {self.current_confidence:.1%}", True, confidence_color)
            self.screen.blit(confidence_text, (10, 50))
            
            instruction_text = self.font.render("Move your hand UP/DOWN to control the bird!", True, WHITE)
            self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 90))
        
        # Controls
        controls_text = self.font.render("C: toggle camera | R: reset | SPACE: restart | ESC: quit", True, WHITE)
        self.screen.blit(controls_text, (10, SCREEN_HEIGHT - 60))
        
        if not self.learning_background and self.current_confidence < 0.3:
            tip_text = self.font.render("TIP: Move your hand in front of the camera!", True, YELLOW)
            self.screen.blit(tip_text, (10, SCREEN_HEIGHT - 30))
        
        if self.game_over:
            game_over_text = self.big_font.render("GAME OVER", True, RED)
            restart_text = self.font.render("Press SPACE to restart or ESC to quit", True, WHITE)
            
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
            
    def restart_game(self) -> None:
        self.bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.pipe_timer = 0
        
    def reset_tracker(self) -> None:
        self.hand_tracker = BetterHandTracker()
        self.learning_background = True
        print("🔄 Resetting hand tracking...")
        
    def run(self) -> None:
        running: bool = True
        
        print("🎮 Better Hand Gesture Flappy Bird!")
        print("📋 How it works:")
        print("1. First, keep your hands OUT of camera view for background learning")
        print("2. Once learned, move your hand in front of camera to control bird")
        print("3. Hand UP = Bird UP, Hand DOWN = Bird DOWN")
        print("\n⌨️  Controls:")
        print("- C: Toggle camera view")
        print("- R: Reset hand tracking")
        print("- SPACE: Restart when game over")
        print("- ESC: Quit")
        
        while running:
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
                        self.reset_tracker()
                        
            if not self.game_over and not self.learning_background:
                hand_y = self.get_hand_control()
                self.bird.update(hand_y, self.current_confidence)
                
                self.pipe_timer += 1
                if self.pipe_timer > 120:
                    self.spawn_pipe()
                    self.pipe_timer = 0
                    
                self.update_pipes()
                
                if self.bird.y <= 0 or self.bird.y >= SCREEN_HEIGHT:
                    self.game_over = True
            else:
                self.get_hand_control()
                    
            self.screen.fill(BLUE)
            self.bird.draw(self.screen)
            
            for pipe in self.pipes:
                pipe.draw(self.screen)
                
            self.draw_hud()
            pygame.display.flip()
            self.clock.tick(FPS)
            
        self.camera.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run() 