import pygame
import cv2
import numpy as np
import random
from typing import Tuple, List, Optional
import math

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

class ImprovedHandTracker:
    def __init__(self):
        self.calibrated: bool = False
        self.skin_lower_hsv: np.ndarray = np.array([0, 20, 70])
        self.skin_upper_hsv: np.ndarray = np.array([20, 255, 255])
        self.skin_lower_ycrcb: np.ndarray = np.array([0, 133, 77])
        self.skin_upper_ycrcb: np.ndarray = np.array([255, 173, 127])
        self.calibration_frames: int = 0
        self.calibration_samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.hand_positions_history: List[Tuple[float, float]] = []
        self.detection_confidence: float = 0.0
        
        # Background subtractor for better detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.background_learned: bool = False
        self.frame_count: int = 0
        
    def learn_background(self, frame: np.ndarray) -> None:
        """Learn background for 30 frames"""
        if self.frame_count < 30:
            self.bg_subtractor.apply(frame, learningRate=0.5)
            self.frame_count += 1
        else:
            self.background_learned = True
            
    def calibrate_skin_color(self, frame: np.ndarray, center_x: int, center_y: int) -> None:
        """Improved calibration using multiple color spaces"""
        sample_size: int = 40
        x1: int = max(0, center_x - sample_size)
        x2: int = min(frame.shape[1], center_x + sample_size)
        y1: int = max(0, center_y - sample_size)
        y2: int = min(frame.shape[0], center_y + sample_size)
        
        sample_region = frame[y1:y2, x1:x2]
        
        # Convert to different color spaces
        hsv_sample = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
        ycrcb_sample = cv2.cvtColor(sample_region, cv2.COLOR_BGR2YCrCb)
        
        # Calculate mean values
        mean_hsv = np.mean(hsv_sample.reshape(-1, 3), axis=0)
        mean_ycrcb = np.mean(ycrcb_sample.reshape(-1, 3), axis=0)
        
        self.calibration_samples.append((mean_hsv, mean_ycrcb))
        self.calibration_frames += 1
        
        if self.calibration_frames >= 15:  # More samples for better calibration
            # Calculate average values
            avg_hsv = np.mean([sample[0] for sample in self.calibration_samples], axis=0)
            avg_ycrcb = np.mean([sample[1] for sample in self.calibration_samples], axis=0)
            
            # Set HSV range with better margins
            h_range: int = 15
            s_range: int = 80
            v_range: int = 80
            
            self.skin_lower_hsv = np.array([
                max(0, avg_hsv[0] - h_range),
                max(30, avg_hsv[1] - s_range),
                max(60, avg_hsv[2] - v_range)
            ])
            
            self.skin_upper_hsv = np.array([
                min(179, avg_hsv[0] + h_range),
                255,
                255
            ])
            
            # Set YCrCb range
            self.skin_lower_ycrcb = np.array([
                max(0, avg_ycrcb[0] - 40),
                max(77, avg_ycrcb[1] - 20),
                max(133, avg_ycrcb[2] - 20)
            ])
            
            self.skin_upper_ycrcb = np.array([
                min(255, avg_ycrcb[0] + 40),
                min(173, avg_ycrcb[1] + 20),
                min(127, avg_ycrcb[2] + 20)
            ])
            
            self.calibrated = True
            print(f"✅ Calibration complete!")
            print(f"HSV range: {self.skin_lower_hsv} - {self.skin_upper_hsv}")
            print(f"YCrCb range: {self.skin_lower_ycrcb} - {self.skin_upper_ycrcb}")
    
    def create_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create improved skin mask using multiple color spaces"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Create masks for both color spaces
        mask_hsv = cv2.inRange(hsv, self.skin_lower_hsv, self.skin_upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, self.skin_lower_ycrcb, self.skin_upper_ycrcb)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Apply morphological operations to clean up the mask
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        # Fill holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)
        # Smooth edges
        combined_mask = cv2.medianBlur(combined_mask, 5)
        
        return combined_mask
    
    def get_hand_position(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], np.ndarray, float]:
        """Get hand position with improved detection and return debug info"""
        if not self.calibrated:
            return None, frame, 0.0
        
        # Learn background if not learned yet
        if not self.background_learned:
            self.learn_background(frame)
            return None, frame, 0.0
            
        # Create skin mask
        skin_mask = self.create_skin_mask(frame)
        
        # Use background subtraction to help with detection
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.01)
        
        # Combine skin and foreground masks
        combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create debug frame
        debug_frame = frame.copy()
        
        # Draw masks on debug frame
        mask_colored = cv2.applyColorMap(skin_mask, cv2.COLORMAP_JET)
        debug_frame = cv2.addWeighted(debug_frame, 0.7, mask_colored, 0.3, 0)
        
        best_hand_pos = None
        confidence = 0.0
        
        if contours:
            # Filter contours by area and shape
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Minimum area for hand
                    # Check if contour is roughly hand-shaped
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Hand should have reasonable solidity (not too complex)
                    if 0.3 < solidity < 0.9:
                        valid_contours.append((contour, area, solidity))
            
            if valid_contours:
                # Sort by area and take the largest valid contour
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                best_contour, area, solidity = valid_contours[0]
                
                # Calculate confidence based on area and solidity
                confidence = min(1.0, (area / 10000) * solidity)
                
                # Get the highest point of the contour (fingertips area)
                highest_point = tuple(best_contour[best_contour[:, :, 1].argmin()][0])
                
                # Get center of mass for stability
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Use weighted average of highest point and center
                    final_x = int(0.3 * highest_point[0] + 0.7 * cx)
                    final_y = int(0.7 * highest_point[1] + 0.3 * cy)
                    
                    # Draw detection on debug frame
                    cv2.drawContours(debug_frame, [best_contour], -1, (0, 255, 0), 2)
                    cv2.circle(debug_frame, (final_x, final_y), 10, (255, 0, 0), -1)
                    cv2.circle(debug_frame, highest_point, 5, (0, 0, 255), -1)
                    
                    # Normalize coordinates
                    x_norm = final_x / frame.shape[1]
                    y_norm = final_y / frame.shape[0]
                    
                    best_hand_pos = (x_norm, y_norm)
        
        # Smooth hand position using history
        if best_hand_pos:
            self.hand_positions_history.append(best_hand_pos)
            if len(self.hand_positions_history) > 5:
                self.hand_positions_history.pop(0)
            
            # Calculate smoothed position
            if len(self.hand_positions_history) >= 3:
                avg_x = sum(pos[0] for pos in self.hand_positions_history) / len(self.hand_positions_history)
                avg_y = sum(pos[1] for pos in self.hand_positions_history) / len(self.hand_positions_history)
                best_hand_pos = (avg_x, avg_y)
        
        self.detection_confidence = confidence
        
        # Add confidence and position info to debug frame
        cv2.putText(debug_frame, f"Confidence: {confidence:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if best_hand_pos:
            cv2.putText(debug_frame, f"Position: {best_hand_pos[0]:.2f}, {best_hand_pos[1]:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return best_hand_pos, debug_frame, confidence

class Bird:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.velocity: float = 0
        self.size: int = BIRD_SIZE
        self.target_y: int = y
        self.last_hand_y: Optional[float] = None
        
    def update(self, hand_y: Optional[float] = None, confidence: float = 0.0) -> None:
        """Update bird position with improved hand tracking"""
        if hand_y is not None and confidence > 0.3:  # Only use confident detections
            # Convert hand y (0-1) to screen coordinates (inverted)
            target_y = int((1 - hand_y) * SCREEN_HEIGHT)
            
            # Apply deadzone in the middle to reduce jitter
            screen_center = SCREEN_HEIGHT // 2
            deadzone = 50
            
            if abs(target_y - screen_center) < deadzone:
                target_y = screen_center
            
            self.target_y = target_y
            
            # Smooth movement with confidence-based speed
            diff: float = self.target_y - self.y
            speed_factor = 0.1 + (confidence * 0.1)  # Faster movement with higher confidence
            self.y += int(diff * speed_factor)
            
            # Reset velocity when hand is detected with good confidence
            self.velocity = 0
            self.last_hand_y = hand_y
            
        else:
            # Apply gravity when no confident hand detected
            self.velocity += GRAVITY
            self.y += int(self.velocity)
            
        # Keep bird on screen
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))
        
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the bird with confidence indicator"""
        # Main bird body
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.size)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.size, 3)
        
        # Draw eye
        eye_x: int = self.x + 10
        eye_y: int = self.y - 5
        pygame.draw.circle(screen, BLACK, (eye_x, eye_y), 5)
        
        # Draw beak
        beak_points = [
            (self.x + self.size - 5, self.y),
            (self.x + self.size + 10, self.y - 3),
            (self.x + self.size + 10, self.y + 3)
        ]
        pygame.draw.polygon(screen, (255, 165, 0), beak_points)
        
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
        pygame.display.set_caption("Improved Hand Gesture Flappy Bird")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        # Game objects
        self.bird: Bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes: List[Pipe] = []
        self.score: int = 0
        self.game_over: bool = False
        
        # Hand tracking
        self.hand_tracker: ImprovedHandTracker = ImprovedHandTracker()
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
        self.current_confidence: float = 0.0
        
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
                cv2.circle(frame, (center_x, center_y), 40, (0, 255, 0), 3)
                cv2.putText(frame, "Place hand in GREEN circle", (center_x - 120, center_y - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Calibrating... {self.hand_tracker.calibration_frames}/15", 
                           (center_x - 130, center_y + 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Keep hand still!", (center_x - 80, center_y + 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Auto-calibrate from center region
                self.hand_tracker.calibrate_skin_color(frame, center_x, center_y)
            else:
                self.calibration_mode = False
                print("🎮 Calibration complete! Starting game...")
                print("✋ Move your hand up and down to control the bird!")
        
        # Get hand position and debug info
        hand_pos, debug_frame, confidence = self.hand_tracker.get_hand_position(frame)
        self.current_confidence = confidence
        
        # Show camera feed with debug info
        if self.show_camera:
            if self.calibration_mode:
                cv2.imshow('Hand Tracking - Calibration', frame)
            else:
                cv2.imshow('Hand Tracking - Debug View (Press C to toggle)', debug_frame)
            cv2.waitKey(1)
        
        # Return hand y position
        if hand_pos and confidence > 0.2:
            return hand_pos[1]
        
        return None
        
    def draw_hud(self) -> None:
        """Draw enhanced game HUD"""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Confidence indicator
        confidence_color = GREEN if self.current_confidence > 0.5 else (YELLOW if self.current_confidence > 0.3 else RED)
        confidence_text = self.font.render(f"Detection: {self.current_confidence:.1%}", True, confidence_color)
        self.screen.blit(confidence_text, (10, 50))
        
        # Instructions
        if self.calibration_mode:
            instruction_text = self.font.render("Calibrating... Place hand in green circle and keep still", True, WHITE)
        else:
            instruction_text = self.font.render("Move your hand UP/DOWN to control the bird!", True, WHITE)
        self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 90))
        
        # Controls
        controls_text = self.font.render("C: toggle camera | R: recalibrate | SPACE: restart | ESC: quit", True, WHITE)
        self.screen.blit(controls_text, (10, SCREEN_HEIGHT - 60))
        
        # Tips
        if not self.calibration_mode and self.current_confidence < 0.3:
            tip_text = self.font.render("TIP: Make sure your hand is well-lit and visible!", True, YELLOW)
            self.screen.blit(tip_text, (10, SCREEN_HEIGHT - 30))
        
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
        self.hand_tracker = ImprovedHandTracker()
        self.calibration_mode = True
        print("🔄 Recalibrating hand detection...")
        
    def run(self) -> None:
        """Main game loop"""
        running: bool = True
        
        print("🎮 Improved Hand Gesture Flappy Bird!")
        print("📋 Instructions:")
        print("1. First, calibrate by placing your hand in the green circle")
        print("2. Keep your hand still during calibration")
        print("3. Once calibrated, move your hand up/down to control the bird")
        print("4. Make sure you have good lighting on your hand")
        print("\n⌨️  Controls:")
        print("- C: Toggle camera view")
        print("- R: Recalibrate")
        print("- SPACE: Restart when game over")
        print("- ESC: Quit")
        
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
                self.bird.update(hand_y, self.current_confidence)
                
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