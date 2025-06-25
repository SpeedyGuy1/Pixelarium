# pixelarium.py

import pygame
import numpy as np
import sys
import json
import os
from scipy.signal import convolve2d
from pygame import gfxdraw

# === Constants & Helpers ===

DEFAULT_WIDTH, DEFAULT_HEIGHT = 960, 540
DEFAULT_CELL_SIZE = 6
FPS = 30

# Patterns dictionary: simple ones + more can be added
PATTERNS = {
    "glider": np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=np.uint8),
    "pulsar": np.array([
        [0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0],
    ], dtype=np.uint8)
}

# === Utility Functions ===

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def blend_color(c1, c2, alpha):
    """Blend colors c1 and c2 by alpha [0..1]."""
    return tuple(int(c1[i] * (1-alpha) + c2[i] * alpha) for i in range(3))


# === Core Pixelarium Class ===

class Pixelarium:
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, cell_size=DEFAULT_CELL_SIZE, states=3):
        pygame.init()
        pygame.mixer.init()

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_w = width // cell_size
        self.grid_h = height // cell_size
        self.states = states

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Pixelarium")
        self.clock = pygame.time.Clock()

        # Grid stores cell states (int from 0 to states-1)
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        self.age_grid = np.zeros_like(self.grid, dtype=np.uint8)  # aging tracker

        # Default colors for each state (can be customized)
        self.color_map = [
            (0,0,0),
            (255,100,100),
            (255,255,100)
        ]
        self.max_age = 15  # max aging frames before full fade

        # Default rules (Conway's Life)
        self.birth = [3]
        self.survival = [2,3]

        # Zoom and pan state
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Undo/redo stacks (store grids)
        self.undo_stack = []
        self.redo_stack = []

        # Sound setup
        self.sound_on = True
        self.sound_freq_base = 220
        self.sound_volume = 0.05
        self.sound_channel = pygame.mixer.Channel(0)
        self.sound_length_ms = 50

        # Font for HUD
        self.font = pygame.font.SysFont("Consolas", 18)

        # Mouse drawing
        self.mouse_drawing = False
        self.draw_state = 1  # state to draw on click

        # Pattern selection
        self.current_pattern = None

        # Flags
        self.running = True

    # --- Grid Access Helpers ---

    def in_bounds(self, x, y):
        return 0 <= x < self.grid_w and 0 <= y < self.grid_h

    # --- Rules ---

    def count_neighbors(self):
        # Conway's life neighbors count
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
        return convolve2d((self.grid > 0).astype(np.uint8), kernel, mode='same', boundary='wrap')

    def update_grid(self):
        neighbors = self.count_neighbors()
        new_grid = np.zeros_like(self.grid)

        # Apply birth/survival rules for state=1 (alive)
        birth_mask = (self.grid == 0) & np.isin(neighbors, self.birth)
        survive_mask = (self.grid == 1) & np.isin(neighbors, self.survival)

        new_grid[birth_mask] = 1
        new_grid[survive_mask] = 1

        # Aging: increment age of alive cells, reset dead
        self.age_grid[new_grid == 1] = np.minimum(self.age_grid[new_grid == 1] + 1, self.max_age)
        self.age_grid[new_grid == 0] = 0

        self.grid = new_grid

        # Play sound based on number of live cells
        if self.sound_on:
            live_cells = np.sum(new_grid == 1)
            if live_cells > 0 and not self.sound_channel.get_busy():
                freq = clamp(self.sound_freq_base + live_cells * 5, 220, 880)
                self.play_tone(freq)

    # --- Drawing ---

    def draw_grid(self):
        self.screen.fill((0,0,0))

        for y in range(self.grid_h):
            for x in range(self.grid_w):
                state = self.grid[y,x]
                if state == 0:
                    continue
                age = self.age_grid[y,x]
                base_color = self.color_map[state] if state < len(self.color_map) else (255,255,255)
                # Fade older cells to black
                alpha = age / self.max_age
                color = blend_color(base_color, (0,0,0), 1-alpha)
                rect = pygame.Rect(
                    x * self.cell_size * self.zoom + self.offset_x,
                    y * self.cell_size * self.zoom + self.offset_y,
                    self.cell_size * self.zoom,
                    self.cell_size * self.zoom
                )
                # Draw a filled rect with color
                pygame.draw.rect(self.screen, color, rect)

    def draw_hud(self):
        lines = [
            f"Pixelarium | FPS: {int(self.clock.get_fps())} | Zoom: {self.zoom:.2f}",
            f"Draw state: {self.draw_state} | Mouse drag drawing: {'ON' if self.mouse_drawing else 'OFF'}",
            "Controls: Q=Quit | R=Reset | +/- Zoom | Mouse drag to draw",
            "1=Glider, 2=Pulsar (spawn pattern at mouse)",
            "Z=Undo | Y=Redo | S=Save | L=Load | M=Toggle Sound",
            "Arrow keys = Pan"
        ]
        y = 5
        for line in lines:
            surf = self.font.render(line, True, (180,180,180))
            self.screen.blit(surf, (5, y))
            y += 22

    # --- Input ---

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
                elif event.key == pygame.K_1:
                    self.spawn_pattern("glider")
                elif event.key == pygame.K_2:
                    self.spawn_pattern("pulsar")
                elif event.key == pygame.K_UP:
                    self.offset_y += 20
                elif event.key == pygame.K_DOWN:
                    self.offset_y -= 20
                elif event.key == pygame.K_LEFT:
                    self.offset_x += 20
                elif event.key == pygame.K_RIGHT:
                    self.offset_x -= 20

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_drawing = True
                elif event.button == 4:  # scroll up zoom in
                    self.zoom = min(self.zoom + 0.1, 10)
                elif event.button == 5:  # scroll down zoom out
                    self.zoom = max(self.zoom - 0.1, 0.1)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_drawing = False

        if self.mouse_drawing:
            self.handle_mouse()

    # --- Undo / Redo ---

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

    # --- Save / Load ---

    def save_grid(self, filename="pixelarium_save.json"):
        data = {
            "width": self.grid_w,
            "height": self.grid_h,
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
        if data["width"] != self.grid_w or data["height"] != self.grid_h:
            print("Save grid size does not match current grid size.")
            return
        self.grid = np.array(data["grid"], dtype=np.uint8)
        self.age_grid = np.zeros_like(self.grid)
        print(f"Loaded grid from {filename}")

    # --- Patterns ---

    def spawn_pattern(self, pattern_name):
        mx, my = pygame.mouse.get_pos()
        gx = int((mx - self.offset_x) / (self.cell_size * self.zoom))
        gy = int((my - self.offset_y) / (self.cell_size * self.zoom))
        pattern = PATTERNS.get(pattern_name)
        if pattern is None:
            print(f"Pattern {pattern_name} not found.")
            return
        ph, pw = pattern.shape
        self.save_undo()
        for y in range(ph):
            for x in range(pw):
                px = gx + x
                py = gy + y
                if self.in_bounds(px, py):
                    self.grid[py, px] = pattern[y,x]
                    self.age_grid[py, px] = 0

    # --- Sound ---

    def play_tone(self, frequency):
        # Generate a short sine wave sound at given frequency
        sample_rate = 44100
        length = int(sample_rate * (self.sound_length_ms / 1000))
        buf = (np.sin(2 * np.pi * np.arange(length) * frequency / sample_rate) * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(buf)
        sound.set_volume(self.sound_volume)
        self.sound_channel.play(sound)

    # --- Main loop ---

    def run(self):
        while self.running:
            self.handle_events()
            self.update_grid()
            self.draw_grid()
            self.draw_hud()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()
