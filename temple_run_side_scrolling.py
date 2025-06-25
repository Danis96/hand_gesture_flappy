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
SCREEN_WIDTH: int = 1200
SCREEN_HEIGHT: int = 800
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
PURPLE: Tuple[int, int, int] = (128, 0, 128)

# Game settings - Side-scrolling 2.5D
INITIAL_SPEED: float = 4.0
MAX_SPEED: float = 12.0
SPEED_INCREMENT: float = 0.03
LANES: int = 3  # Top, Middle, Bottom lanes
LANE_HEIGHT: int = 150
CHARACTER_X: int = 200  # Fixed X position for character
GROUND_Y: int = SCREEN_HEIGHT - 100

# Lane Y positions (from top to bottom)
LANE_POSITIONS: List[int] = [
    GROUND_Y - LANE_HEIGHT * 2,  # Top lane
    GROUND_Y - LANE_HEIGHT,      # Middle lane  
    GROUND_Y                     # Bottom lane (ground)
]

class CharacterState(Enum):
    RUNNING = 1
    JUMPING = 2
    SLIDING = 3
    BOOSTING = 4

class ObstacleType(Enum):
    BARRIER = 1
    GAP = 2
    LOW_BARRIER = 3
    HANGING_OBSTACLE = 4

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
        """Detect multiple hand gestures for side-scrolling control"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        debug_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        gestures = {
            'lane_direction': 0,  # -1 up, 0 stay, 1 down
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
                    
                    hands_data.append({
                        'center': (center_x, center_y),
                        'landmarks': landmarks
                    })
            
            gestures['hands_detected'] = len(hands_data)
            
            if len(hands_data) == 1:
                # Single hand control
                hand = hands_data[0]
                center_x, center_y = hand['center']
                
                # Lane control (up/down based on hand position)
                if center_y < frame.shape[0] * 0.25:  # Hand high up
                    gestures['lane_direction'] = -1  # Move to upper lane
                    gestures['vertical_action'] = 'jump'
                    cv2.putText(debug_frame, "JUMP UP!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif center_y > frame.shape[0] * 0.75:  # Hand low down
                    gestures['lane_direction'] = 1   # Move to lower lane
                    gestures['vertical_action'] = 'slide'
                    cv2.putText(debug_frame, "SLIDE DOWN!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif center_y < frame.shape[0] * 0.4:  # Medium high
                    gestures['vertical_action'] = 'jump'
                    cv2.putText(debug_frame, "JUMP", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif center_y > frame.shape[0] * 0.6:  # Medium low
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
                    cv2.putText(debug_frame, "POWER BOOST!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
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
        self.x: int = x  # Fixed X position in side-scrolling
        self.y: int = y
        self.lane: int = 2  # 0=top, 1=middle, 2=bottom (start on ground)
        self.target_lane: int = 2
        self.state: CharacterState = CharacterState.RUNNING
        self.jump_height: float = 0
        self.jump_velocity: float = 0
        self.slide_timer: int = 0
        self.boost_timer: int = 0
        self.invincible_timer: int = 0
        self.animation_frame: float = 0
        
        # Character dimensions
        self.width: int = 40
        self.height: int = 70
        
    def update(self, gestures: Dict[str, any]) -> None:
        """Update character based on gestures"""
        # Lane movement (vertical in side-scrolling)
        if gestures['lane_direction'] == -1 and self.target_lane > 0:
            self.target_lane -= 1  # Move up
        elif gestures['lane_direction'] == 1 and self.target_lane < LANES - 1:
            self.target_lane += 1  # Move down
        
        # Smooth lane transition
        target_y = LANE_POSITIONS[self.target_lane]
        if abs(self.y - target_y) > 5:
            self.y += (target_y - self.y) * 0.12
        else:
            self.y = target_y
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
        
        # Animation
        self.animation_frame += 0.3
    
    def jump(self) -> None:
        """Make character jump"""
        self.state = CharacterState.JUMPING
        self.jump_velocity = -18
    
    def slide(self) -> None:
        """Make character slide"""
        self.state = CharacterState.SLIDING
        self.slide_timer = 50  # 0.83 seconds at 60 FPS
    
    def boost(self) -> None:
        """Activate boost"""
        self.state = CharacterState.BOOSTING
        self.boost_timer = 200  # 3.33 seconds
        self.invincible_timer = 200
    
    def update_state(self) -> None:
        """Update character based on current state"""
        if self.state == CharacterState.JUMPING:
            self.jump_velocity += 1.2  # Gravity
            self.jump_height += self.jump_velocity
            
            if self.jump_height >= 0:
                self.jump_height = 0
                self.jump_velocity = 0
                self.state = CharacterState.RUNNING
        
        elif self.state == CharacterState.BOOSTING:
            if self.boost_timer <= 0:
                self.state = CharacterState.RUNNING
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw character in side-scrolling view"""
        draw_x = self.x
        draw_y = int(self.y + self.jump_height)
        
        # Character body color based on state
        if self.state == CharacterState.BOOSTING or self.invincible_timer > 0:
            body_color = GOLD_COLOR
        elif self.state == CharacterState.SLIDING:
            body_color = BLUE
        else:
            body_color = RED_COLOR
        
        # Draw character based on state
        if self.state == CharacterState.SLIDING:
            # Draw sliding (wider, shorter)
            pygame.draw.ellipse(screen, body_color, 
                              (draw_x - self.width//2, draw_y - self.height//3, 
                               self.width * 1.5, self.height//2))
        else:
            # Draw running character (body + head)
            # Body
            pygame.draw.ellipse(screen, body_color, 
                              (draw_x - self.width//2, draw_y - self.height, 
                               self.width, self.height * 0.7))
            # Head
            pygame.draw.circle(screen, body_color, 
                             (draw_x, draw_y - self.height + 15), self.width//3)
            
            # Running animation (simple leg movement)
            leg_offset = math.sin(self.animation_frame) * 10
            pygame.draw.ellipse(screen, body_color,
                              (draw_x - 10 + leg_offset, draw_y - 20, 8, 20))
            pygame.draw.ellipse(screen, body_color,
                              (draw_x + 2 - leg_offset, draw_y - 20, 8, 20))
        
        # Draw state indicators
        if self.state == CharacterState.BOOSTING:
            # Boost aura
            for i in range(3):
                pygame.draw.circle(screen, GOLD_COLOR, (draw_x, draw_y - self.height//2), 
                                 self.width + i * 10, 2)
        
        if self.invincible_timer > 0 and self.invincible_timer % 10 < 5:
            # Flashing effect when invincible
            pygame.draw.circle(screen, WHITE, (draw_x, draw_y - self.height//2), 
                             self.width + 5, 3)

class Obstacle:
    def __init__(self, x: float, lane: int, obstacle_type: ObstacleType):
        self.x: float = x
        self.lane: int = lane
        self.type: ObstacleType = obstacle_type
        self.width: int = 60
        self.height: int = 70 if obstacle_type != ObstacleType.LOW_BARRIER else 35
        
    def update(self, speed: float) -> None:
        """Move obstacle left in side-scrolling"""
        self.x -= speed
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw obstacle in side-scrolling view"""
        draw_x = int(self.x)
        draw_y = LANE_POSITIONS[self.lane]
        
        if self.type == ObstacleType.BARRIER:
            # Tall barrier
            pygame.draw.rect(screen, STONE_COLOR, 
                           (draw_x - self.width//2, draw_y - self.height, 
                            self.width, self.height))
            pygame.draw.rect(screen, BLACK, 
                           (draw_x - self.width//2, draw_y - self.height, 
                            self.width, self.height), 3)
            
        elif self.type == ObstacleType.LOW_BARRIER:
            # Low barrier (can jump over or slide under)
            pygame.draw.rect(screen, WALL_COLOR, 
                           (draw_x - self.width//2, draw_y - self.height, 
                            self.width, self.height))
            pygame.draw.rect(screen, BLACK, 
                           (draw_x - self.width//2, draw_y - self.height, 
                            self.width, self.height), 3)
            
        elif self.type == ObstacleType.GAP:
            # Gap in the floor (only for bottom lane)
            if self.lane == 2:  # Bottom lane only
                pygame.draw.rect(screen, BLACK, 
                               (draw_x - self.width, draw_y - 10, self.width * 2, 20))
                
        elif self.type == ObstacleType.HANGING_OBSTACLE:
            # Hanging obstacle from ceiling (top lane)
            pygame.draw.rect(screen, PURPLE, 
                           (draw_x - self.width//2, draw_y - self.height - 100, 
                            self.width, self.height))
            pygame.draw.rect(screen, BLACK, 
                           (draw_x - self.width//2, draw_y - self.height - 100, 
                            self.width, self.height), 3)
    
    def get_collision_rect(self) -> pygame.Rect:
        """Get collision rectangle"""
        draw_x = int(self.x)
        draw_y = LANE_POSITIONS[self.lane]
        
        if self.type == ObstacleType.HANGING_OBSTACLE:
            return pygame.Rect(draw_x - self.width//2, draw_y - self.height - 100, 
                             self.width, self.height)
        else:
            return pygame.Rect(draw_x - self.width//2, draw_y - self.height, 
                             self.width, self.height)

class Coin:
    def __init__(self, x: float, lane: int, height_offset: float = 0):
        self.x: float = x
        self.lane: int = lane
        self.height_offset: float = height_offset
        self.collected: bool = False
        self.animation: float = 0
        
    def update(self, speed: float) -> None:
        """Move coin left in side-scrolling"""
        self.x -= speed
        self.animation += 0.25
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw coin in side-scrolling view"""
        if self.collected:
            return
            
        draw_x = int(self.x)
        draw_y = int(LANE_POSITIONS[self.lane] - 40 - self.height_offset - 
                    math.sin(self.animation) * 15)
        
        # Animated spinning coin
        scale = abs(math.cos(self.animation * 0.5)) * 0.5 + 0.5
        radius = int(20 * scale)
        
        pygame.draw.circle(screen, GOLD_COLOR, (draw_x, draw_y), radius)
        pygame.draw.circle(screen, BLACK, (draw_x, draw_y), radius, 2)
        
        # Shine effect
        if scale > 0.8:
            pygame.draw.circle(screen, WHITE, (draw_x - 5, draw_y - 5), 5)
    
    def get_collision_rect(self) -> pygame.Rect:
        """Get collision rectangle"""
        draw_x = int(self.x)
        draw_y = int(LANE_POSITIONS[self.lane] - 40 - self.height_offset)
        return pygame.Rect(draw_x - 20, draw_y - 20, 40, 40)

class TempleRun2_5D:
    def __init__(self):
        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🏛️ TEMPLE RUN 2.5D - Side-Scrolling Hand Control 🏛️")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        
        # Game objects
        self.character: Character = Character(CHARACTER_X, LANE_POSITIONS[2])
        self.obstacles: List[Obstacle] = []
        self.coins: List[Coin] = []
        
        # Game state
        self.speed: float = INITIAL_SPEED
        self.distance: int = 0
        self.score: int = 0
        self.coins_collected: int = 0
        self.game_over: bool = False
        self.camera_offset: float = 0
        
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
        
        # Choose appropriate obstacle type for lane
        if lane == 0:  # Top lane
            obstacle_types = [ObstacleType.HANGING_OBSTACLE, ObstacleType.LOW_BARRIER]
        elif lane == 1:  # Middle lane
            obstacle_types = [ObstacleType.BARRIER, ObstacleType.LOW_BARRIER]
        else:  # Bottom lane
            obstacle_types = [ObstacleType.BARRIER, ObstacleType.GAP, ObstacleType.LOW_BARRIER]
            
        obstacle_type = random.choice(obstacle_types)
        spawn_x = SCREEN_WIDTH + random.randint(100, 300)
        
        self.obstacles.append(Obstacle(spawn_x, lane, obstacle_type))
    
    def spawn_coin(self) -> None:
        """Spawn coin"""
        lane = random.randint(0, LANES - 1)
        spawn_x = SCREEN_WIDTH + random.randint(50, 200)
        height_offset = random.choice([0, 30, 60])  # Different heights
        
        self.coins.append(Coin(spawn_x, lane, height_offset))
    
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
            if coin.x < -50:
                self.coins.remove(coin)
        
        # Spawn new objects
        self.obstacle_timer += 1
        if self.obstacle_timer > max(30, 90 - int(self.speed * 5)):
            self.spawn_obstacle()
            self.obstacle_timer = 0
        
        self.coin_timer += 1
        if self.coin_timer > 60:
            self.spawn_coin()
            self.coin_timer = 0
        
        # Increase speed gradually
        if self.speed < MAX_SPEED:
            self.speed += SPEED_INCREMENT
        
        # Update distance and score
        self.distance += int(self.speed)
        self.score = self.distance + self.coins_collected * 15
        
        # Camera shake effect during boost
        if self.character.boost_timer > 0:
            self.camera_offset = math.sin(pygame.time.get_ticks() * 0.3) * 3
        else:
            self.camera_offset = 0
    
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
            if obstacle.x < CHARACTER_X + 50 and obstacle.x > CHARACTER_X - 50:
                if obstacle.lane == self.character.lane:
                    if self.character.invincible_timer <= 0:
                        collision = False
                        
                        if obstacle.type == ObstacleType.BARRIER:
                            collision = True
                        elif obstacle.type == ObstacleType.LOW_BARRIER:
                            if (self.character.state != CharacterState.SLIDING and 
                                self.character.jump_height > -50):
                                collision = True
                        elif obstacle.type == ObstacleType.GAP:
                            if self.character.jump_height > -30:
                                collision = True
                        elif obstacle.type == ObstacleType.HANGING_OBSTACLE:
                            if self.character.jump_height < -30:
                                collision = True
                        
                        if collision:
                            self.game_over = True
        
        # Check coin collisions
        for coin in self.coins:
            if not coin.collected and coin.x < CHARACTER_X + 40 and coin.x > CHARACTER_X - 40:
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
            cv2.imshow('Temple Run 2.5D - Hand Control (Press C to toggle)', debug_frame)
            cv2.waitKey(1)
        
        return gestures
    
    def draw_environment(self) -> None:
        """Draw side-scrolling 2.5D environment"""
        # Apply camera shake
        screen_offset = int(self.camera_offset)
        
        # Sky gradient
        for y in range(0, SCREEN_HEIGHT//2):
            color_intensity = int(100 + (y / (SCREEN_HEIGHT//2)) * 155)
            color = (color_intensity // 4, color_intensity // 3, color_intensity)
            pygame.draw.line(self.screen, color, (0, y + screen_offset), (SCREEN_WIDTH, y + screen_offset))
        
        # Background mountains/temples (parallax effect)
        mountain_offset = int(self.distance * 0.1) % 400
        for i in range(3):
            x = i * 400 - mountain_offset
            points = [(x, SCREEN_HEIGHT//2), (x + 150, SCREEN_HEIGHT//3), (x + 300, SCREEN_HEIGHT//2)]
            pygame.draw.polygon(self.screen, (80, 60, 40), points)
        
        # Ground lanes
        for i, lane_y in enumerate(LANE_POSITIONS):
            # Lane platform
            lane_color = GROUND_COLOR if i == 2 else WALL_COLOR
            pygame.draw.rect(self.screen, lane_color, 
                           (0, lane_y + screen_offset, SCREEN_WIDTH, 30))
            pygame.draw.line(self.screen, BLACK, 
                           (0, lane_y + screen_offset), (SCREEN_WIDTH, lane_y + screen_offset), 2)
        
        # Temple walls and pillars (moving background)
        pillar_offset = int(self.distance * 0.3) % 200
        for i in range(8):
            x = i * 200 - pillar_offset
            # Pillars
            pygame.draw.rect(self.screen, STONE_COLOR, 
                           (x, LANE_POSITIONS[0] - 150 + screen_offset, 30, 150))
            pygame.draw.rect(self.screen, BLACK, 
                           (x, LANE_POSITIONS[0] - 150 + screen_offset, 30, 150), 2)
    
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
        
        # Character state and lane
        lane_names = ["TOP", "MIDDLE", "BOTTOM"]
        state_text = f"Lane: {lane_names[self.character.lane]} | State: {self.character.state.name}"
        if self.character.boost_timer > 0:
            state_text += f" | BOOST: {self.character.boost_timer//60 + 1}s"
        state_surface = self.font.render(state_text, True, GOLD_COLOR if self.character.boost_timer > 0 else WHITE)
        self.screen.blit(state_surface, (10, 170))
        
        # Controls help
        control_texts = [
            "🏃‍♂️ TEMPLE RUN 2.5D - SIDE SCROLLING:",
            "👆 Hand HIGH UP = Jump to Upper Lane",
            "👇 Hand LOW DOWN = Slide to Lower Lane", 
            "👆 Hand UP = Jump",
            "👇 Hand DOWN = Slide",
            "🙌 Both Hands UP = POWER BOOST!",
            "C: Toggle Camera | ESC: Quit"
        ]
        
        for i, text in enumerate(control_texts):
            color = GOLD_COLOR if i == 0 else WHITE
            control_surface = self.font.render(text, True, color)
            self.screen.blit(control_surface, (SCREEN_WIDTH - 420, 10 + i * 25))
        
        # Lane indicators
        for i, lane_y in enumerate(LANE_POSITIONS):
            color = GOLD_COLOR if i == self.character.lane else WHITE
            pygame.draw.circle(self.screen, color, (SCREEN_WIDTH - 50, lane_y), 8)
        
        if self.game_over:
            # Game over screen
            game_over_text = self.big_font.render("TEMPLE COLLAPSED!", True, RED_COLOR)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.font.render("Press SPACE to restart or ESC to quit", True, WHITE)
            
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            
            pygame.draw.rect(self.screen, BLACK, game_over_rect.inflate(20, 20))
            pygame.draw.rect(self.screen, BLACK, score_rect.inflate(20, 20))
            pygame.draw.rect(self.screen, BLACK, restart_rect.inflate(20, 20))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(final_score_text, score_rect)
            self.screen.blit(restart_text, restart_rect)
    
    def restart_game(self) -> None:
        """Restart the game"""
        self.character = Character(CHARACTER_X, LANE_POSITIONS[2])
        self.obstacles = []
        self.coins = []
        self.speed = INITIAL_SPEED
        self.distance = 0
        self.score = 0
        self.coins_collected = 0
        self.game_over = False
        self.obstacle_timer = 0
        self.coin_timer = 0
        self.camera_offset = 0
    
    def run(self) -> None:
        """Main game loop"""
        running: bool = True
        
        print("🏛️ TEMPLE RUN 2.5D - Side-Scrolling Edition! 🏛️")
        print("📋 Instructions:")
        print("🏃‍♂️ Run through the temple from a side view!")
        print("👆 HIGH hand = Jump to UPPER lane")
        print("👇 LOW hand = Slide to LOWER lane")  
        print("👆 Medium HIGH = Regular jump")
        print("👇 Medium LOW = Regular slide")
        print("🙌 Both hands UP = POWER BOOST!")
        print("\n🎮 Three Lanes:")
        print("- TOP: Avoid hanging obstacles")
        print("- MIDDLE: Mixed obstacles")
        print("- BOTTOM: Ground level with gaps")
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
            self.draw_environment()
            
            # Draw game objects
            for obstacle in self.obstacles:
                obstacle.draw(self.screen)
            
            for coin in self.coins:
                coin.draw(self.screen)
            
            # Draw character
            self.character.draw(self.screen)
            
            # Draw HUD
            self.draw_hud()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = TempleRun2_5D()
    game.run() 