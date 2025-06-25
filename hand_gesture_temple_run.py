import pygame
import cv2
import mediapipe as mp
import numpy as np
import random
import math
from typing import Tuple, List, Optional, Dict
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH: int = 1000
SCREEN_HEIGHT: int = 700
FPS: int = 60

# Colors - Temple Theme
GROUND_COLOR: Tuple[int, int, int] = (139, 69, 19)  # Brown
WALL_COLOR: Tuple[int, int, int] = (160, 82, 45)    # Saddle brown
STONE_COLOR: Tuple[int, int, int] = (105, 105, 105)  # Gray
GOLD_COLOR: Tuple[int, int, int] = (255, 215, 0)    # Gold
GREEN_COLOR: Tuple[int, int, int] = (34, 139, 34)   # Forest green
RED_COLOR: Tuple[int, int, int] = (255, 0, 0)       # Red
WHITE: Tuple[int, int, int] = (255, 255, 255)
BLACK: Tuple[int, int, int] = (0, 0, 0)
BLUE: Tuple[int, int, int] = (0, 100, 255)

# Game settings
INITIAL_SPEED: float = 2.0
MAX_SPEED: float = 8.0
SPEED_INCREMENT: float = 0.02
LANES: int = 3
LANE_WIDTH: int = 150
PERSPECTIVE_FACTOR: float = 0.8

class CharacterState(Enum):
    RUNNING = 1
    JUMPING = 2
    SLIDING = 3
    BOOSTING = 4

class ObstacleType(Enum):
    BARRIER = 1
    GAP = 2
    LOW_BARRIER = 3

class PowerUpType(Enum):
    SPEED_BOOST = 1
    INVINCIBILITY = 2
    COIN_MAGNET = 3

class AdvancedHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def get_hand_gestures(self, frame: np.ndarray) -> Tuple[Dict[str, any], np.ndarray]:
        """Detect multiple hand gestures"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        debug_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        gestures = {
            'lane_direction': 0,  # -1 left, 0 center, 1 right
            'vertical_action': 'none',  # 'jump', 'slide', 'none'
            'boost': False,
            'hands_detected': 0
        }
        
        if results.multi_hand_landmarks:
            hands_data = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(debug_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get hand position data
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * frame.shape[1]
                    y = landmark.y * frame.shape[0]
                    landmarks.append((x, y))
                
                if len(landmarks) > 8:
                    # Calculate hand center
                    center_x = sum(lm[0] for lm in landmarks) / len(landmarks)
                    center_y = sum(lm[1] for lm in landmarks) / len(landmarks)
                    
                    # Get key landmarks
                    wrist = landmarks[0]
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]
                    
                    hands_data.append({
                        'center': (center_x, center_y),
                        'wrist': wrist,
                        'index_tip': index_tip,
                        'middle_tip': middle_tip,
                        'landmarks': landmarks
                    })
            
            gestures['hands_detected'] = len(hands_data)
            
            if len(hands_data) == 1:
                # Single hand control
                hand = hands_data[0]
                center_x, center_y = hand['center']
                
                # Lane control (left/right based on hand position)
                if center_x < frame.shape[1] * 0.33:
                    gestures['lane_direction'] = -1  # Left
                    cv2.putText(debug_frame, "MOVE LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif center_x > frame.shape[1] * 0.67:
                    gestures['lane_direction'] = 1   # Right
                    cv2.putText(debug_frame, "MOVE RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Vertical actions (jump/slide based on hand height)
                if center_y < frame.shape[0] * 0.3:
                    gestures['vertical_action'] = 'jump'
                    cv2.putText(debug_frame, "JUMP", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif center_y > frame.shape[0] * 0.7:
                    gestures['vertical_action'] = 'slide'
                    cv2.putText(debug_frame, "SLIDE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw hand center
                cv2.circle(debug_frame, (int(center_x), int(center_y)), 15, (255, 255, 0), -1)
                
            elif len(hands_data) == 2:
                # Two hands - check for boost gesture
                hand1, hand2 = hands_data
                y1 = hand1['center'][1]
                y2 = hand2['center'][1]
                
                # Both hands up = boost
                if y1 < frame.shape[0] * 0.4 and y2 < frame.shape[0] * 0.4:
                    gestures['boost'] = True
                    cv2.putText(debug_frame, "BOOST!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Draw both hand centers
                cv2.circle(debug_frame, (int(hand1['center'][0]), int(hand1['center'][1])), 15, (255, 255, 0), -1)
                cv2.circle(debug_frame, (int(hand2['center'][0]), int(hand2['center'][1])), 15, (255, 255, 0), -1)
        
        # Add gesture info to debug frame
        cv2.putText(debug_frame, f"Hands: {gestures['hands_detected']}", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Lane: {gestures['lane_direction']}", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return gestures, debug_frame

class Character:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.lane: int = 1  # 0=left, 1=center, 2=right
        self.target_lane: int = 1
        self.state: CharacterState = CharacterState.RUNNING
        self.jump_height: float = 0
        self.jump_velocity: float = 0
        self.slide_timer: int = 0
        self.boost_timer: int = 0
        self.invincible_timer: int = 0
        
        # Character dimensions
        self.width: int = 30
        self.height: int = 60
        
    def update(self, gestures: Dict[str, any]) -> None:
        """Update character based on gestures"""
        # Lane movement
        if gestures['lane_direction'] == -1 and self.target_lane > 0:
            self.target_lane -= 1
        elif gestures['lane_direction'] == 1 and self.target_lane < LANES - 1:
            self.target_lane += 1
        
        # Smooth lane transition
        target_x = SCREEN_WIDTH // 2 - LANE_WIDTH + self.target_lane * LANE_WIDTH
        if abs(self.x - target_x) > 5:
            self.x += (target_x - self.x) * 0.15
        else:
            self.x = target_x
            self.lane = self.target_lane
        
        # Vertical actions
        if gestures['vertical_action'] == 'jump' and self.state == CharacterState.RUNNING:
            self.jump()
        elif gestures['vertical_action'] == 'slide' and self.state == CharacterState.RUNNING:
            self.slide()
        
        # Boost
        if gestures['boost'] and self.boost_timer <= 0:
            self.boost()
        
        # Update state-specific logic
        self.update_state()
        
        # Update timers
        if self.boost_timer > 0:
            self.boost_timer -= 1
        if self.invincible_timer > 0:
            self.invincible_timer -= 1
        if self.slide_timer > 0:
            self.slide_timer -= 1
            if self.slide_timer <= 0:
                self.state = CharacterState.RUNNING
    
    def jump(self) -> None:
        """Make character jump"""
        self.state = CharacterState.JUMPING
        self.jump_velocity = -15
    
    def slide(self) -> None:
        """Make character slide"""
        self.state = CharacterState.SLIDING
        self.slide_timer = 45  # 0.75 seconds at 60 FPS
    
    def boost(self) -> None:
        """Activate boost"""
        self.state = CharacterState.BOOSTING
        self.boost_timer = 180  # 3 seconds
        self.invincible_timer = 180
    
    def update_state(self) -> None:
        """Update character based on current state"""
        if self.state == CharacterState.JUMPING:
            self.jump_velocity += 1  # Gravity
            self.jump_height += self.jump_velocity
            
            if self.jump_height >= 0:
                self.jump_height = 0
                self.jump_velocity = 0
                self.state = CharacterState.RUNNING
        
        elif self.state == CharacterState.BOOSTING:
            if self.boost_timer <= 0:
                self.state = CharacterState.RUNNING
    
    def draw(self, screen: pygame.Surface, camera_offset: float) -> None:
        """Draw character with 3D perspective"""
        # Calculate 3D position
        perspective_scale = 1.0 - (camera_offset * 0.1)
        draw_x = int(self.x)
        draw_y = int(self.y + self.jump_height)
        
        # Character body color based on state
        if self.state == CharacterState.BOOSTING or self.invincible_timer > 0:
            body_color = GOLD_COLOR
        elif self.state == CharacterState.SLIDING:
            body_color = BLUE
        else:
            body_color = RED_COLOR
        
        # Draw character
        if self.state == CharacterState.SLIDING:
            # Draw sliding (rectangle)
            pygame.draw.rect(screen, body_color, 
                           (draw_x - self.width//2, draw_y, self.width, self.height//2))
        else:
            # Draw standing (circle + rectangle)
            pygame.draw.circle(screen, body_color, (draw_x, draw_y - self.height//2), self.width//2)
            pygame.draw.rect(screen, body_color, 
                           (draw_x - self.width//2, draw_y - self.height//2, self.width, self.height))
        
        # Draw state indicator
        if self.state == CharacterState.BOOSTING:
            pygame.draw.circle(screen, GOLD_COLOR, (draw_x, draw_y - self.height), 20, 3)

class Obstacle:
    def __init__(self, x: float, lane: int, obstacle_type: ObstacleType):
        self.x: float = x
        self.lane: int = lane
        self.type: ObstacleType = obstacle_type
        self.width: int = 50
        self.height: int = 60 if obstacle_type != ObstacleType.LOW_BARRIER else 30
        
    def update(self, speed: float) -> None:
        """Move obstacle towards player"""
        self.x -= speed
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw obstacle with 3D perspective"""
        # Calculate perspective
        distance = max(0.1, self.x / SCREEN_HEIGHT)
        scale = 1.0 / (distance * PERSPECTIVE_FACTOR + 1)
        
        draw_x = int(SCREEN_WIDTH // 2 + (self.lane - 1) * LANE_WIDTH * scale)
        draw_y = int(SCREEN_HEIGHT - 200 + (1 - scale) * 100)
        draw_width = int(self.width * scale)
        draw_height = int(self.height * scale)
        
        if self.type == ObstacleType.BARRIER:
            pygame.draw.rect(screen, STONE_COLOR, (draw_x - draw_width//2, draw_y - draw_height, draw_width, draw_height))
            pygame.draw.rect(screen, BLACK, (draw_x - draw_width//2, draw_y - draw_height, draw_width, draw_height), 2)
        elif self.type == ObstacleType.LOW_BARRIER:
            pygame.draw.rect(screen, WALL_COLOR, (draw_x - draw_width//2, draw_y - draw_height, draw_width, draw_height))
            pygame.draw.rect(screen, BLACK, (draw_x - draw_width//2, draw_y - draw_height, draw_width, draw_height), 2)
        elif self.type == ObstacleType.GAP:
            # Draw gap edges
            pygame.draw.rect(screen, BLACK, (draw_x - draw_width, draw_y - 10, draw_width//2, 20))
            pygame.draw.rect(screen, BLACK, (draw_x + draw_width//2, draw_y - 10, draw_width//2, 20))
    
    def get_collision_rect(self, character_y: int) -> pygame.Rect:
        """Get collision rectangle"""
        distance = max(0.1, self.x / SCREEN_HEIGHT)
        scale = 1.0 / (distance * PERSPECTIVE_FACTOR + 1)
        
        draw_x = int(SCREEN_WIDTH // 2 + (self.lane - 1) * LANE_WIDTH * scale)
        draw_y = int(SCREEN_HEIGHT - 200 + (1 - scale) * 100)
        draw_width = int(self.width * scale)
        draw_height = int(self.height * scale)
        
        return pygame.Rect(draw_x - draw_width//2, draw_y - draw_height, draw_width, draw_height)

class Coin:
    def __init__(self, x: float, lane: int, y: float = 0):
        self.x: float = x
        self.lane: int = lane
        self.y: float = y  # Height for floating coins
        self.collected: bool = False
        self.animation: float = 0
        
    def update(self, speed: float) -> None:
        """Move coin towards player"""
        self.x -= speed
        self.animation += 0.2
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw coin with 3D perspective"""
        if self.collected:
            return
            
        distance = max(0.1, self.x / SCREEN_HEIGHT)
        scale = 1.0 / (distance * PERSPECTIVE_FACTOR + 1)
        
        draw_x = int(SCREEN_WIDTH // 2 + (self.lane - 1) * LANE_WIDTH * scale)
        draw_y = int(SCREEN_HEIGHT - 200 + (1 - scale) * 100 - self.y - math.sin(self.animation) * 10)
        radius = int(15 * scale)
        
        pygame.draw.circle(screen, GOLD_COLOR, (draw_x, draw_y), radius)
        pygame.draw.circle(screen, BLACK, (draw_x, draw_y), radius, 2)
    
    def get_collision_rect(self) -> pygame.Rect:
        """Get collision rectangle"""
        distance = max(0.1, self.x / SCREEN_HEIGHT)
        scale = 1.0 / (distance * PERSPECTIVE_FACTOR + 1)
        
        draw_x = int(SCREEN_WIDTH // 2 + (self.lane - 1) * LANE_WIDTH * scale)
        draw_y = int(SCREEN_HEIGHT - 200 + (1 - scale) * 100 - self.y)
        radius = int(15 * scale)
        
        return pygame.Rect(draw_x - radius, draw_y - radius, radius * 2, radius * 2)

class TempleRunGame:
    def __init__(self):
        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🏛️ TEMPLE RUN - Hand Gesture Control 🏛️")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        # Game objects
        self.character: Character = Character(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)
        self.obstacles: List[Obstacle] = []
        self.coins: List[Coin] = []
        self.powerups: List[Dict] = []
        
        # Game state
        self.speed: float = INITIAL_SPEED
        self.distance: int = 0
        self.score: int = 0
        self.coins_collected: int = 0
        self.game_over: bool = False
        
        # Hand tracking
        self.hand_tracker: AdvancedHandTracker = AdvancedHandTracker()
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Fonts
        self.font: pygame.font.Font = pygame.font.Font(None, 36)
        self.big_font: pygame.font.Font = pygame.font.Font(None, 72)
        
        # Spawn timers
        self.obstacle_timer: int = 0
        self.coin_timer: int = 0
        
        self.show_camera: bool = True
        
    def spawn_obstacle(self) -> None:
        """Spawn random obstacle"""
        lane = random.randint(0, LANES - 1)
        obstacle_type = random.choice(list(ObstacleType))
        spawn_distance = SCREEN_HEIGHT + random.randint(200, 500)
        
        self.obstacles.append(Obstacle(spawn_distance, lane, obstacle_type))
    
    def spawn_coin(self) -> None:
        """Spawn coin"""
        lane = random.randint(0, LANES - 1)
        spawn_distance = SCREEN_HEIGHT + random.randint(100, 300)
        height = random.choice([0, 30, 60])  # Ground, low, high
        
        self.coins.append(Coin(spawn_distance, lane, height))
    
    def update_game_objects(self) -> None:
        """Update all game objects"""
        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update(self.speed)
            if obstacle.x < -100:
                self.obstacles.remove(obstacle)
        
        # Update coins
        for coin in self.coins[:]:
            coin.update(self.speed)
            if coin.x < -100:
                self.coins.remove(coin)
        
        # Spawn new objects
        self.obstacle_timer += 1
        if self.obstacle_timer > 120 - int(self.speed * 10):  # Faster spawning as speed increases
            self.spawn_obstacle()
            self.obstacle_timer = 0
        
        self.coin_timer += 1
        if self.coin_timer > 80:
            self.spawn_coin()
            self.coin_timer = 0
        
        # Increase speed gradually
        if self.speed < MAX_SPEED:
            self.speed += SPEED_INCREMENT
        
        # Update distance and score
        self.distance += int(self.speed)
        self.score = self.distance + self.coins_collected * 10
    
    def check_collisions(self) -> None:
        """Check for collisions"""
        character_rect = pygame.Rect(
            self.character.x - self.character.width//2,
            self.character.y - self.character.height + self.character.jump_height,
            self.character.width,
            self.character.height
        )
        
        # Check obstacle collisions
        for obstacle in self.obstacles:
            if obstacle.x < 100 and obstacle.x > -50:  # Near character
                if obstacle.lane == self.character.lane:
                    if self.character.invincible_timer <= 0:
                        # Check specific collision based on obstacle type and character state
                        collision = False
                        
                        if obstacle.type == ObstacleType.BARRIER:
                            if self.character.state != CharacterState.JUMPING or self.character.jump_height > -40:
                                collision = True
                        elif obstacle.type == ObstacleType.LOW_BARRIER:
                            if self.character.state != CharacterState.SLIDING and self.character.state != CharacterState.JUMPING:
                                collision = True
                        elif obstacle.type == ObstacleType.GAP:
                            if self.character.state != CharacterState.JUMPING:
                                collision = True
                        
                        if collision:
                            self.game_over = True
        
        # Check coin collisions
        for coin in self.coins:
            if not coin.collected and coin.x < 100 and coin.x > -50:
                if coin.lane == self.character.lane:
                    coin_rect = coin.get_collision_rect()
                    if character_rect.colliderect(coin_rect):
                        coin.collected = True
                        self.coins_collected += 1
    
    def get_hand_control(self) -> Dict[str, any]:
        """Get hand gestures"""
        ret, frame = self.camera.read()
        if not ret:
            return {'lane_direction': 0, 'vertical_action': 'none', 'boost': False, 'hands_detected': 0}
            
        frame = cv2.flip(frame, 1)
        gestures, debug_frame = self.hand_tracker.get_hand_gestures(frame)
        
        if self.show_camera:
            cv2.imshow('Temple Run - Hand Control (Press C to toggle)', debug_frame)
            cv2.waitKey(1)
        
        return gestures
    
    def draw_3d_environment(self) -> None:
        """Draw 3D temple environment"""
        # Sky gradient
        for y in range(0, SCREEN_HEIGHT//2):
            color_intensity = int(135 + (y / (SCREEN_HEIGHT//2)) * 120)
            color = (color_intensity // 3, color_intensity // 2, color_intensity)
            pygame.draw.line(self.screen, color, (0, y), (SCREEN_WIDTH, y))
        
        # Ground with perspective
        ground_points = [
            (0, SCREEN_HEIGHT),
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 200),
            (100, SCREEN_HEIGHT - 200)
        ]
        pygame.draw.polygon(self.screen, GROUND_COLOR, ground_points)
        
        # Lane dividers
        for i in range(1, LANES):
            start_x = SCREEN_WIDTH // 2 - LANE_WIDTH + i * LANE_WIDTH
            end_x = SCREEN_WIDTH // 2 - 100 + i * 100
            pygame.draw.line(self.screen, BLACK, (start_x, SCREEN_HEIGHT), (end_x, SCREEN_HEIGHT - 200), 3)
        
        # Temple walls (perspective)
        left_wall = [
            (0, SCREEN_HEIGHT),
            (0, SCREEN_HEIGHT//2),
            (100, SCREEN_HEIGHT - 200),
            (100, SCREEN_HEIGHT)
        ]
        right_wall = [
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            (SCREEN_WIDTH, SCREEN_HEIGHT//2),
            (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 200),
            (SCREEN_WIDTH - 100, SCREEN_HEIGHT)
        ]
        
        pygame.draw.polygon(self.screen, WALL_COLOR, left_wall)
        pygame.draw.polygon(self.screen, WALL_COLOR, right_wall)
        pygame.draw.polygon(self.screen, BLACK, left_wall, 3)
        pygame.draw.polygon(self.screen, BLACK, right_wall, 3)
    
    def draw_hud(self) -> None:
        """Draw game HUD"""
        # Score and stats
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        distance_text = self.font.render(f"Distance: {self.distance}m", True, WHITE)
        coins_text = self.font.render(f"Coins: {self.coins_collected}", True, GOLD_COLOR)
        speed_text = self.font.render(f"Speed: {self.speed:.1f}", True, WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(distance_text, (10, 50))
        self.screen.blit(coins_text, (10, 90))
        self.screen.blit(speed_text, (10, 130))
        
        # Character state
        state_text = f"State: {self.character.state.name}"
        if self.character.boost_timer > 0:
            state_text += f" (BOOST: {self.character.boost_timer//60 + 1}s)"
        state_surface = self.font.render(state_text, True, GOLD_COLOR if self.character.boost_timer > 0 else WHITE)
        self.screen.blit(state_surface, (10, 170))
        
        # Controls help
        control_texts = [
            "🏃‍♂️ TEMPLE RUN HAND CONTROLS:",
            "✋ Left/Right Hand = Change Lanes",
            "👆 Hand UP = Jump",
            "👇 Hand DOWN = Slide", 
            "🙌 Both Hands UP = BOOST!",
            "C: Toggle Camera | ESC: Quit"
        ]
        
        for i, text in enumerate(control_texts):
            color = GOLD_COLOR if i == 0 else WHITE
            control_surface = self.font.render(text, True, color)
            self.screen.blit(control_surface, (SCREEN_WIDTH - 350, 10 + i * 30))
        
        if self.game_over:
            # Game over screen
            game_over_text = self.big_font.render("TEMPLE COLLAPSED!", True, RED_COLOR)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.font.render("Press SPACE to restart or ESC to quit", True, WHITE)
            
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(final_score_text, score_rect)
            self.screen.blit(restart_text, restart_rect)
    
    def restart_game(self) -> None:
        """Restart the game"""
        self.character = Character(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)
        self.obstacles = []
        self.coins = []
        self.speed = INITIAL_SPEED
        self.distance = 0
        self.score = 0
        self.coins_collected = 0
        self.game_over = False
        self.obstacle_timer = 0
        self.coin_timer = 0
    
    def run(self) -> None:
        """Main game loop"""
        running: bool = True
        
        print("🏛️ TEMPLE RUN - Hand Gesture Edition! 🏛️")
        print("📋 Instructions:")
        print("🏃‍♂️ Run through the ancient temple using hand gestures!")
        print("✋ Move hand LEFT/RIGHT to change lanes")
        print("👆 Move hand UP to JUMP over obstacles")  
        print("👇 Move hand DOWN to SLIDE under barriers")
        print("🙌 Both hands UP for POWER BOOST!")
        print("\n⌨️ Controls:")
        print("- C: Toggle camera view")
        print("- SPACE: Restart when game over")
        print("- ESC: Quit")
        print("\n🎯 Collect coins and avoid obstacles!")
        print("🚀 Speed increases as you progress!")
        
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
                # Get hand gestures
                gestures = self.get_hand_control()
                
                # Update character
                self.character.update(gestures)
                
                # Update game
                self.update_game_objects()
                self.check_collisions()
            else:
                # Still show camera when game over
                self.get_hand_control()
            
            # Draw everything
            self.draw_3d_environment()
            
            # Draw game objects
            for obstacle in self.obstacles:
                obstacle.draw(self.screen)
            
            for coin in self.coins:
                coin.draw(self.screen)
            
            # Draw character
            self.character.draw(self.screen, 0)
            
            # Draw HUD
            self.draw_hud()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = TempleRunGame()
    game.run() 