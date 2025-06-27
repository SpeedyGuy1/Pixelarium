import pygame
from pixelarium import Pixelarium

brush_size = 3  # Default pen size

# --- Material behavior ---
def sand_behavior(grid, x, y):
    if y + 1 >= grid.shape[0]:
        return
    if grid[y + 1, x] == 0:
        grid[y + 1, x], grid[y, x] = grid[y, x], 0
    elif x > 0 and grid[y + 1, x - 1] == 0:
        grid[y + 1, x - 1], grid[y, x] = grid[y, x], 0
    elif x + 1 < grid.shape[1] and grid[y + 1, x + 1] == 0:
        grid[y + 1, x + 1], grid[y, x] = grid[y, x], 0

# --- Initialize game ---
game = Pixelarium(cell_size=4, states=2, title="Falling Sand (Pixelarium v4)")
game.register_material(
    value=1,
    color=(194, 178, 128),
    behavior=sand_behavior
)
game.draw_state = 1

# --- Custom brush-drawing ---
def custom_handle_mouse():
    mx, my = pygame.mouse.get_pos()
    gx = int((mx - game.offset_x) / (game.cell_size * game.zoom))
    gy = int((my - game.offset_y) / (game.cell_size * game.zoom))
    radius = brush_size // 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = gx + dx
            y = gy + dy
            if game.in_bounds(x, y):
                game.grid[y, x] = game.draw_state
                game.age_grid[y, x] = 0
game.handle_mouse = custom_handle_mouse

# --- Custom input handler ---
original_handle_events = game.handle_events
def custom_handle_events():
    global brush_size
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFTBRACKET:
                brush_size = max(1, brush_size - 1)
                print(f"Brush size: {brush_size}")
            elif event.key == pygame.K_RIGHTBRACKET:
                brush_size = min(15, brush_size + 1)
                print(f"Brush size: {brush_size}")
        # Forward all events to the original handler
        pygame.event.post(event)
    original_handle_events()
game.handle_events = custom_handle_events

# --- Run the simulation ---
game.run()
