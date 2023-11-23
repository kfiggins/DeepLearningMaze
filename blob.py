import pygame
import random
import math

from utils import line_intersection

class Blob:
    def __init__(self, x, y, history_length=5, num_sensors=8, sensor_range=100):
        self.x = x
        self.y = y
        self.radius = 10
        self.color = (0, 128, 255)  # Blue color
        self.alive = True
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.movement_history = [(x, y)] * history_length
        self.points_scored = 0

        # Sensor attributes
        self.num_sensors = num_sensors
        self.sensor_range = sensor_range
        self.sensor_angles = [i * (360 / num_sensors) for i in range(num_sensors)]
        self.sensor_data = [sensor_range] * num_sensors  # Initialize with max range

    def update_position(self, x, y):
        self.movement_history.pop(0)
        self.movement_history.append((x, y))
        self.x = x
        self.y = y

    def update_sensor_data(self, walls):
        for i, angle in enumerate(self.sensor_angles):
            sensor_x = self.x + math.cos(math.radians(angle)) * self.sensor_range
            sensor_y = self.y + math.sin(math.radians(angle)) * self.sensor_range
            self.sensor_data[i] = self.calculate_distance_to_wall(sensor_x, sensor_y, walls)

    def calculate_distance_to_wall(self, sensor_x, sensor_y, walls):
        shortest_distance = self.sensor_range
        sensor_line = (self.x, self.y, sensor_x, sensor_y)

        for wall in walls:
            intersection = line_intersection(sensor_line, wall)
            if intersection:
                distance = math.sqrt((intersection[0] - self.x) ** 2 + (intersection[1] - self.y) ** 2)
                if distance < shortest_distance:
                    shortest_distance = distance

        return shortest_distance


    def draw(self, screen):
        if self.alive:
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
