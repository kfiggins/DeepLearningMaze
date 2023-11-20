import pygame
import random


class Blob:
    def __init__(self, x, y, history_length=5):
        self.x = x
        self.y = y
        self.radius = 10
        self.color = (0, 128, 255)  # Blue color
        self.alive = True
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.movement_history = [(x, y)] * history_length
        self.points_scored = 0

    def update_position(self, x, y):
        self.movement_history.pop(0)
        self.movement_history.append((x, y))
        self.x = x
        self.y = y

    def draw(self, screen):
        if self.alive:
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
