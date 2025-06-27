# pixelarium.py v4 with dynamic resize, material rules, temp system, and UI hooks

import pygame
import numpy as np
import sys
import json
import os
from scipy.signal import convolve2d
from pygame import gfxdraw

DEFAULT_WIDTH, DEFAULT_HEIGHT = 960, 540
DEFAULT_CELL_SIZE = 6
FPS = 30

class Tool:
    def __init__(self, name, brush, color, value):
        self.name = name
        self.brush = brush
        self.color = color
        self.value = value

class Material:
    def __init__(self, name, color, behavior):
        self.name = name
        self.color = color
        self.behavior = behavior  # Callable(grid, x, y)

class Pixelarium:
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, cell_size=DEFAULT_CELL_SIZE, states=3,
                 update_func=None, palette=None, tools=None, title="Pixelarium", fps=FPS):
        pygame.init()
        pygame.mixer.init()

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.states = states
        self.fps = fps

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.color_map = palette or [
            (0, 0, 0),
            (255, 100, 100),
            (255, 255, 100)
        ]
        self.max_age = 15

        self.palette = self.color_map
        self.tools = tools or []

        self.birth = [3]
        self.survival = [2, 3]

        self.undo_stack = []
        self.redo_stack = []

        self.sound_on = True
        self.sound_freq_base = 220
        self.sound_volume = 0.05
        self.sound_channel = pygame.mixer.Channel(0)
        self.sound_length_ms = 50

        self.font = pygame.font.SysFont("Consolas", 18)

        self.mouse_drawing = False
        self.draw_state = 1

        self.running = True

        self.update_func = update_func or self.update_default

        self.material_behaviors = {}

        self.resize_grids()

    def resize_grids(self):
        self.grid_w = self.screen.get_width() // self.cell_size
        self.grid_h = self.screen.get_height() // self.cell_size
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        self.temp_grid = np.zeros_like(self.grid, dtype=np.float32)
        self.age_grid = np.zeros_like(self.grid, dtype=np.uint8)

    def register_material(self, value, color, behavior=None):
        self.color_map.append(color)
        self.material_behaviors[value] = behavior

    def update_default(self, grid):
        height, width = grid.shape
        for y in reversed(range(height - 1)):
            for x in range(width):
                val = grid[y][x]
                if val in self.material_behaviors and callable(self.material_behaviors[val]):
                    self.material_behaviors[val](grid, x, y)

    def in_bounds(self, x, y):
        return 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]

    def draw_grid(self):
        self.screen.fill((0,0,0))
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                state = self.grid[y,x]
                if state == 0:
                    continue
                age = self.age_grid[y,x]
                base_color = self.color_map[state] if state < len(self.color_map) else (255,255,255)
                alpha = age / self.max_age
                color = tuple(int(base_color[i] * (1 - alpha)) for i in range(3))
                rect = pygame.Rect(
                    x * self.cell_size * self.zoom + self.offset_x,
                    y * self.cell_size * self.zoom + self.offset_y,
                    self.cell_size * self.zoom,
                    self.cell_size * self.zoom
                )
                pygame.draw.rect(self.screen, color, rect)

    def draw_hud(self):
        lines = [
            f"Pixelarium v4 | FPS: {int(self.clock.get_fps())} | Zoom: {self.zoom:.2f}",
            f"Draw state: {self.draw_state} | Mouse drag: {'ON' if self.mouse_drawing else 'OFF'}",
            "Q=Quit R=Reset +/- Zoom Mouse=Draw Z=Undo Y=Redo",
            "S=Save L=Load M=ToggleSound Arrows=Pan"
        ]
        y = 5
        for line in lines:
            surf = self.font.render(line, True, (180,180,180))
            self.screen.blit(surf, (5, y))
            y += 22

    def handle_mouse(self):
        mx, my = pygame.mouse.get_pos()
        gx = int((mx - self.offset_x) / (self.cell_size * self.zoom))
        gy = int((my - self.offset_y) / (self.cell_size * self.zoom))
        if self.in_bounds(gx, gy):
            if pygame.mouse.get_pressed()[0]:
                self.grid[gy,gx] = self.draw_state
                self.age_grid[gy,gx] = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.resize_grids()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.save_undo()
                    self.grid.fill(0)
                    self.age_grid.fill(0)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom = min(self.zoom + 0.1, 10)
                elif event.key == pygame.K_MINUS:
                    self.zoom = max(self.zoom - 0.1, 0.1)
                elif event.key == pygame.K_z:
                    self.undo()
                elif event.key == pygame.K_y:
                    self.redo()
                elif event.key == pygame.K_s:
                    self.save_grid()
                elif event.key == pygame.K_l:
                    self.load_grid()
                elif event.key == pygame.K_m:
                    self.sound_on = not self.sound_on
                elif event.key == pygame.K_UP:
                    self.offset_y += 20
                elif event.key == pygame.K_DOWN:
                    self.offset_y -= 20
                elif event.key == pygame.K_LEFT:
                    self.offset_x += 20
                elif event.key == pygame.K_RIGHT:
                    self.offset_x -= 20
                elif event.key == pygame.K_LEFTBRACKET:
                    brush_size = max(1, brush_size - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                     brush_size = min(15, brush_size + 1)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_drawing = True
                elif event.button == 4:
                    self.zoom = min(self.zoom + 0.1, 10)
                elif event.button == 5:
                    self.zoom = max(self.zoom - 0.1, 0.1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_drawing = False
        if self.mouse_drawing:
            self.handle_mouse()

    def save_undo(self):
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.grid.copy())
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.grid.copy())
            self.grid = self.undo_stack.pop()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.grid.copy())
            self.grid = self.redo_stack.pop()

    def save_grid(self, filename="pixelarium_save.json"):
        data = {
            "width": self.grid.shape[1],
            "height": self.grid.shape[0],
            "states": self.states,
            "grid": self.grid.tolist()
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Saved grid to {filename}")

    def load_grid(self, filename="pixelarium_save.json"):
        if not os.path.exists(filename):
            print(f"No save file {filename} found!")
            return
        with open(filename, "r") as f:
            data = json.load(f)
        if data["width"] != self.grid.shape[1] or data["height"] != self.grid.shape[0]:
            print("Save grid size does not match current grid size.")
            return
        self.grid = np.array(data["grid"], dtype=np.uint8)
        self.age_grid = np.zeros_like(self.grid)
        print(f"Loaded grid from {filename}")

    def play_tone(self, frequency):
        sample_rate = 44100
        length = int(sample_rate * (self.sound_length_ms / 1000))
        buf = (np.sin(2 * np.pi * np.arange(length) * frequency / sample_rate) * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(buf)
        sound.set_volume(self.sound_volume)
        self.sound_channel.play(sound)

    def run(self):
        while self.running:
            self.handle_events()
            if callable(self.update_func):
                self.update_func(self.grid)
            self.draw_grid()
            self.draw_hud()
            pygame.display.flip()
            self.clock.tick(self.fps)
        pygame.quit()
        sys.exit()
