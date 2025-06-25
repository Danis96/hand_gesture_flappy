import pygame
import cv2
import mediapipe as mp
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

class MediaPipeHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for stability
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_positions_history: List[float] = []
        
    def get_hand_position(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], np.ndarray, float]:
        """Get hand position using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        debug_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        hand_position = None
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    debug_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))
                
                # Use the index finger tip (landmark 8) for control
                if len(landmarks) > 8:
                    index_tip = landmarks[8]
                    
                    # Draw a circle at the tracking point
                    cv2.circle(debug_frame, index_tip, 15, (255, 0, 0), -1)
                    cv2.putText(debug_frame, "INDEX TIP", (index_tip[0] + 20, index_tip[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Normalize coordinates
                    x_norm = index_tip[0] / frame.shape[1]
                    y_norm = index_tip[1] / frame.shape[0]
                    
                    hand_position = (x_norm, y_norm)
                    confidence = 0.9  # High confidence with MediaPipe
                    
                    # Draw tracking info
                    cv2.putText(debug_frame, f"Hand detected!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(debug_frame, f"Position: ({x_norm:.2f}, {y_norm:.2f})", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw hand box
                    x_coords = [lm[0] for lm in landmarks]
                    y_coords = [lm[1] for lm in landmarks]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame.shape[1], x_max + padding)
                    y_max = min(frame.shape[0], y_max + padding)
                    
                    cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                break  # Only process first hand
        else:
            cv2.putText(debug_frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_frame, "Show your hand to the camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return hand_position, debug_frame, confidence
    
    def get_smoothed_y(self, y_pos: float) -> float:
        """Apply smoothing to Y position"""
        self.hand_positions_history.append(y_pos)
        
        # Keep only last 3 positions for responsiveness
        if len(self.hand_positions_history) > 3:
            self.hand_positions_history.pop(0)
        
        # Return weighted average
        if len(self.hand_positions_history) == 1:
            return y_pos
        
        # More weight to recent positions
        weights = [0.2, 0.3, 0.5][-len(self.hand_positions_history):]
        weighted_sum = sum(pos * weight for pos, weight in zip(self.hand_positions_history, weights))
        return weighted_sum / sum(weights)

class Bird:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.velocity: float = 0
        self.size: int = BIRD_SIZE
        
    def update(self, hand_y: Optional[float] = None, confidence: float = 0.0) -> None:
        """Update bird position"""
        if hand_y is not None and confidence > 0.5:
            # Convert hand y to bird position (inverted with some offset)
            # Map hand position more intuitively
            target_y = int((1 - hand_y) * SCREEN_HEIGHT)
            
            # Add some constraints to make it easier to control
            target_y = max(50, min(SCREEN_HEIGHT - 50, target_y))
            
            # Smooth movement
            diff = target_y - self.y
            self.y += int(diff * 0.3)  # More responsive
            
            # Reduce velocity when hand is controlling
            self.velocity *= 0.3
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
        eye_x: int = self.x + 10
        eye_y: int = self.y - 5
        pygame.draw.circle(screen, BLACK, (eye_x, eye_y), 5)
        
        # Beak
        beak_points = [
            (self.x + self.size - 5, self.y),
            (self.x + self.size + 10, self.y - 3),
            (self.x + self.size + 10, self.y + 3)
        ]
        pygame.draw.polygon(screen, (255, 165, 0), beak_points)
        
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
        pygame.display.set_caption("Professional Hand Gesture Flappy Bird - MediaPipe")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        self.bird: Bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes: List[Pipe] = []
        self.score: int = 0
        self.game_over: bool = False
        
        self.hand_tracker: MediaPipeHandTracker = MediaPipeHandTracker()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.font: pygame.font.Font = pygame.font.Font(None, 36)
        self.big_font: pygame.font.Font = pygame.font.Font(None, 72)
        
        self.pipe_timer: int = 0
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
            
        frame = cv2.flip(frame, 1)  # Mirror image
        
        # Get hand position
        hand_pos, debug_frame, confidence = self.hand_tracker.get_hand_position(frame)
        self.current_confidence = confidence
        
        # Show camera feed
        if self.show_camera:
            cv2.imshow('MediaPipe Hand Tracking (Press C to toggle)', debug_frame)
            cv2.waitKey(1)
        
        if hand_pos and confidence > 0.5:
            smoothed_y = self.hand_tracker.get_smoothed_y(hand_pos[1])
            return smoothed_y
        
        return None
        
    def draw_hud(self) -> None:
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Hand detection status
        if self.current_confidence > 0.5:
            status_color = GREEN
            status_text = "Hand Detected ✅"
        else:
            status_color = RED
            status_text = "Show Your Hand ❌"
        
        detection_text = self.font.render(status_text, True, status_color)
        self.screen.blit(detection_text, (10, 50))
        
        # Instructions
        instruction_text = self.font.render("Point with INDEX FINGER - UP/DOWN to control bird!", True, WHITE)
        self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 90))
        
        # Controls
        controls_text = self.font.render("C: toggle camera | SPACE: restart | ESC: quit", True, WHITE)
        self.screen.blit(controls_text, (10, SCREEN_HEIGHT - 60))
        
        # Tips
        if self.current_confidence < 0.5:
            tip_text = self.font.render("TIP: Point your INDEX FINGER clearly at the camera!", True, YELLOW)
            self.screen.blit(tip_text, (10, SCREEN_HEIGHT - 30))
        
        if self.game_over:
            game_over_text = self.big_font.render("GAME OVER", True, RED)
            restart_text = self.font.render("Press SPACE to restart or ESC to quit", True, WHITE)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
            self.screen.blit(final_score_text, score_rect)
            
    def restart_game(self) -> None:
        self.bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.pipe_timer = 0
        
    def run(self) -> None:
        running: bool = True
        
        print("🎮 Professional Hand Gesture Flappy Bird with MediaPipe!")
        print("📋 Instructions:")
        print("1. Point your INDEX FINGER at the camera")
        print("2. Move your finger UP to make bird go UP")
        print("3. Move your finger DOWN to make bird go DOWN")
        print("4. Keep your hand visible and well-lit")
        print("\n⌨️  Controls:")
        print("- C: Toggle camera view")
        print("- SPACE: Restart when game over")
        print("- ESC: Quit")
        print("\n🎯 The game tracks your INDEX FINGER TIP for precise control!")
        
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
                        
            if not self.game_over:
                hand_y = self.get_hand_control()
                self.bird.update(hand_y, self.current_confidence)
                
                self.pipe_timer += 1
                if self.pipe_timer > 120:  # Every 2 seconds
                    self.spawn_pipe()
                    self.pipe_timer = 0
                    
                self.update_pipes()
                
                if self.bird.y <= 0 or self.bird.y >= SCREEN_HEIGHT:
                    self.game_over = True
            else:
                # Still show camera feed when game over
                self.get_hand_control()
                    
            # Draw everything
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