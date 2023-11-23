import pygame
from constants import START_POINT, WALLS, END_POINT, WIDTH, HEIGHT
import torch
import sys
import random
import math

def check_collision(blob, walls):
    blob_rect = pygame.Rect(blob.x - blob.radius, blob.y - blob.radius, 
                            blob.radius * 2, blob.radius * 2)
    for wall in walls:
        if blob_rect.colliderect(wall):
            return True
    return False

def calculate_reward(blob, has_collided):
    end_reward = 100  # Reward for reaching near the end point
    collision_penalty = 50  # Base penalty for collision
    goal_tolerance = 25  # Define a square around the endpoint
    jitter_penalty = 0.75  # Penalty to discourage jittering behavior
    movement_consistency_bonus = 0.2  # Bonus for consistent movement

    current_position = blob.movement_history[-1]
    old_distance = distance(blob.movement_history[0], END_POINT)
    new_distance = distance(current_position, END_POINT)

    # Reward for being within the goal area
    if abs(current_position[0] - END_POINT[0]) <= goal_tolerance and abs(current_position[1] - END_POINT[1]) <= goal_tolerance:
        blob.points_scored += 1
        blob.alive = False
        return end_reward

    # Collision penalty
    if has_collided:
        penalty_reduction_factor = max(0, (1 - new_distance / old_distance))
        return -collision_penalty + penalty_reduction_factor * collision_penalty

    # Analyze movement pattern for jittering
    jittering_detected = is_jittering_detected(blob.movement_history)
    
    # Movement reward/penalty
    if new_distance < old_distance:
        reward = 1 + (movement_consistency_bonus if not jittering_detected else 0)
    elif new_distance > old_distance:
        reward = -1
    else:
        reward = -jitter_penalty  # Apply a small penalty if no significant movement towards the goal

    return reward

def is_jittering_detected(movement_history):
    # Check if the blob is frequently reversing its direction
    direction_changes = 0
    for i in range(1, len(movement_history)):
        if movement_history[i] != movement_history[i-1]:
            direction_changes += 1
    return direction_changes >= len(movement_history) // 2  # True if changes are frequent

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def draw_maze(screen, walls):
    WHITE = (255, 255, 255)

    # Define the thickness of the walls
    wall_thickness = 5

    # Walls of the maze (start_x, start_y, end_x, end_y)

    wall_rects = []
    for wall in walls:
            start_x, start_y, end_x, end_y = wall
            if start_x == end_x:  # Vertical wall
                rect = pygame.Rect(start_x - wall_thickness // 2, start_y, 
                                wall_thickness, end_y - start_y)
            else:  # Horizontal wall
                rect = pygame.Rect(start_x, start_y - wall_thickness // 2, 
                                end_x - start_x, wall_thickness)
            pygame.draw.line(screen, WHITE, wall[:2], wall[2:], wall_thickness)
            wall_rects.append(rect)

    # Draw start and end points
    # pygame.draw.circle(screen, (0, 255, 0), START_POINT, 10)  # Green for start
    pygame.draw.circle(screen, (255, 0, 0), END_POINT, 10)    # Red for end

    return wall_rects

def draw_start_button(screen, button_rect, text):
    button_color = (0, 200, 0)  # Green color
    text_color = (255, 255, 255)  # White color
    

    pygame.draw.rect(screen, button_color, button_rect)
    font = pygame.font.SysFont(None, 40)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)



def show_start_screen(screen, button_rect):
    start = False
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    start = True

        screen.fill((0, 0, 0))  # Clear screen
        draw_start_button(screen, button_rect, "Start Game")
        pygame.display.update()

def get_observation(blob):
    # Normalized position
    normalized_x = blob.x / WIDTH
    normalized_y = blob.y / HEIGHT
    position_data = [normalized_x, normalized_y]

    # Normalized sensor data
    normalized_sensor_data = [distance / blob.sensor_range for distance in blob.sensor_data]

    # Calculate and normalize the direction to the exit
    exit_direction_x = (END_POINT[0] - blob.x) / WIDTH
    exit_direction_y = (END_POINT[1] - blob.y) / HEIGHT
    exit_direction_magnitude = math.sqrt(exit_direction_x ** 2 + exit_direction_y ** 2)
    normalized_exit_direction = [exit_direction_x / exit_direction_magnitude, exit_direction_y / exit_direction_magnitude]

    # Combine all data
    observation = torch.tensor(position_data + normalized_sensor_data + normalized_exit_direction, dtype=torch.float)

    return observation



def move_blob(blob, action):
    step_size = 3
    new_x, new_y = blob.x, blob.y  # Initialize with current position

    if action == 0:  # Left
        new_x -= step_size
    elif action == 1:  # Right
        new_x += step_size
    elif action == 2:  # Up
        new_y -= step_size
    elif action == 3:  # Down
        new_y += step_size

    # Update the position using the update_position method
    blob.update_position(new_x, new_y)



def explode(blob, screen):
    explosion_color = (255, 0, 0)  # Red
    pygame.draw.circle(screen, explosion_color, (blob.x, blob.y), blob.radius * 2)
    pygame.display.update()
    pygame.time.delay(5)  

def reset_blobs(blobs):
    for blob in blobs:
        blob.x = 200  # Initial x position
        blob.y = 200  # Initial y position
        blob.alive = True  # Reset the alive status


epsilon = 0.1  # Probability of random action

def select_action(net, observation):
    if random.random() < epsilon:
        # Choose a random action
        return random.randint(0, 3)
    else:
        # Use the network to decide the action
        output = net(observation)
        return torch.argmax(output).item()


def determine_best_blob(blobs):
    # Initialize variables to store the best blob and the minimum distance
    best_blob_index = None
    min_distance = float('inf')

    for idx, blob in enumerate(blobs):
        blob_position = (blob.x, blob.y)
        current_distance = distance(blob_position, END_POINT)
        if current_distance < min_distance:
            min_distance = current_distance
            best_blob_index = idx

    return best_blob_index



def evaluate_blob_performance(blob):
    # Calculate the blob's distance to the end point
    blob_position = (blob.x, blob.y)
    current_distance = distance(blob_position, END_POINT)

    # Return the negative distance as the performance score
    # Smaller distances (closer to the end point) result in higher scores
    return -current_distance

# After a certain number of episodes, average the weights of the best performing blobs
def average_best_performers(nets, blobs, top_percentage=0.05):
    # Rank blobs first by points scored, then by distance to the end point
    blobs_ranked = sorted(blobs, key=lambda blob: (-blob.points_scored, distance((blob.x, blob.y), END_POINT)))

    # Select the top performers
    top_indices = [blobs.index(blob) for blob in blobs_ranked[:int(len(blobs) * top_percentage)]]

    average_weights = None
    for idx in top_indices:
        net_weights = nets[idx].state_dict()
        if average_weights is None:
            average_weights = {k: v.clone() for k, v in net_weights.items()}
        else:
            for k, v in net_weights.items():
                average_weights[k] += v

    for k in average_weights.keys():
        average_weights[k] /= len(top_indices)

    return average_weights


def line_intersection(line1, line2):
    # Unpack the line segment coordinates
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate denominators
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Lines are parallel

    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    # Check if intersection is within line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Return the intersection point
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    return None