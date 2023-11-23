import pygame
import sys
from blob import Blob
from blob_net import BlobNet
from utils import average_best_performers, calculate_reward, check_collision, draw_maze, explode, get_observation, move_blob, randomize_end_point, reset_blobs, select_action, show_start_screen
from constants import NUMBER_OF_BLOBS, WIDTH, HEIGHT, WALLS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def main():
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze Game")
    font = pygame.font.Font(None, 36)  # Default system font, size 30
    
    total_episodes = 500  # Define the number of episodes

    # Create blob objects for each blob in the game
    blobs = [Blob(200, 200) for _ in range(NUMBER_OF_BLOBS)]

    # Initialize neural networks, one for each blob
    nets = [BlobNet(13, 10, 10, 4) for _ in range(NUMBER_OF_BLOBS)]

    # Set up optimizers for each network (for training)
    optimizers = [optim.Adam(net.parameters(), lr=0.005) for net in nets]

    # Learning rate schedulers to adjust the learning rate over time
    # schedulers = [lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) for optimizer in optimizers]

    button_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 25, 200, 50)  # Position and size
    # Set up game state
    game_state = "running"

    clock = pygame.time.Clock()
    max_episode_duration = 7000  # Maximum time limit for each episode, in milliseconds
    
    # Initialize success count for each blob
    # success_counts = [0] * NUMBER_OF_BLOBS

    for episode in range(total_episodes):
        if episode == 0:
            # Show start screen only before the first episode
            show_start_screen(screen, button_rect)
        END_POINT = randomize_end_point()
        start_time = pygame.time.get_ticks()  # Get the current time
        game_state = "running"
        clock = pygame.time.Clock()

        # Main game loop
        while game_state == "running":
            current_time = pygame.time.get_ticks()
            if current_time - start_time > max_episode_duration:
                game_state = "game over"
                break  # End the episode if the time limit is exceeded

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            screen.fill((0, 0, 0))
            episode_text = font.render(f'Episode: {episode + 1}', True, (255, 255, 255))
            screen.blit(episode_text, (10, 10))  # Adjust position as needed

            # Calculate total exits from all blobs
            total_exits = sum(blob.points_scored for blob in blobs)

            # Display total exits
            total_exits_text = font.render(f'Total Exits: {total_exits}', True, (255, 255, 255))
            screen.blit(total_exits_text, (WIDTH - 200, 10))  # Adjust position as needed

            wall_rects = draw_maze(screen, WALLS, END_POINT)

            all_dead = all(not blob.alive for blob in blobs)
            if all_dead:
                game_state = "game over"
                break  # Break out of the game loop to start a new episode

            # [Game logic for moving blobs, checking collisions, etc.]
            # Loop through each blob for this frame
            for blob, net, optimizer in zip(blobs, nets, optimizers):
                if not blob.alive:
                    continue  # Skip to the next blob if this one is dead
                old_position = (blob.x, blob.y)
                blob.update_sensor_data(WALLS)
                observation = get_observation(blob, END_POINT)
                output = net(observation)

                # Decide action
                action = select_action(net, observation)

                move_blob(blob, action)
                has_collided = check_collision(blob, wall_rects)
                if has_collided:
                    explode(blob, screen)
                    blob.alive = False
                # Calculate reward
                new_position = (blob.x, blob.y)
                reward = calculate_reward(blob, has_collided, END_POINT)
                
                # Convert reward to a tensor
                reward_tensor = torch.tensor([reward], dtype=torch.float)

                # Perform the training step (backpropagation)
                optimizer.zero_grad()
                # Assume the output is a probability distribution and use a suitable loss function
                loss_function = nn.MSELoss()
                loss = loss_function(output, reward_tensor)
                loss.backward()
                optimizer.step()

                blob.draw(screen)

                # Inside your main game loop, after drawing blobs
                # Inside your main game loop, after drawing blobs
                # Sort blobs by points_scored in descending order
                sorted_blobs = sorted(blobs, key=lambda blob: blob.points_scored, reverse=True)

                for idx, blob in enumerate(sorted_blobs):
                    # Draw a small dot with the blob's color
                    pygame.draw.circle(screen, blob.color, (30, 50 + idx * 30), 10)

                    # Display the blob's score
                    points_text = font.render(f': {blob.points_scored}', True, (255, 255, 255))
                    screen.blit(points_text, (50, 30 + idx * 30))  # Adjust position as needed


            pygame.display.update()
            clock.tick(60)


        
        if episode % 10 == 0:  # Averaging after every 10 episodes
            averaged_weights = average_best_performers(nets, blobs, END_POINT)
            for net in nets:
                net.load_state_dict(averaged_weights)

        # Step the schedulers after each episode
        # for scheduler in schedulers:
        #     scheduler.step()
        reset_blobs(blobs)  # Reset blobs for the next episode


if __name__ == "__main__":
    main()
