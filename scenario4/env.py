import pygame


class GridWorldMDP:
    def __init__(self, grid_size=5, decay=0.99):
        self.grid_size = grid_size
        self.decay = decay
        self.state = [0, 0]  # Starting position
        self.time_step = 0

        self.state_size = grid_size ** 2
        self.action_size = 5
        self.time_size = 20
        
        # Reward locations (will alternate)
        self.reward_locations = [
            [1, 1],  # Bottom right
            [3, 3]  # Top right
        ]
        self.current_reward_location = self.reward_locations[0]

    def reset(self):
        self.state = [0, 0]
        self.time_step = 0
        return self.state, self.time_step

    def step(self, action):
        # Update time step
        self.time_step += 1

        # Actions: 0-Noop, 0-Left, 1-Right, 2-Down, 3-Up
        moves = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Calculate new state
        new_x = self.state[0] + moves[action][0]
        new_y = self.state[1] + moves[action][1]
        
        # Boundary check
        new_x = max(0, min(new_x, self.grid_size - 1))
        new_y = max(0, min(new_y, self.grid_size - 1))
        
        self.state = [new_x, new_y]
        
        # Compute reward
        move_penalty = -0.1 if action != 0 else 0
        
        # Change reward location every 10 time steps
        reward_index = (self.time_step // (self.time_size//2)) % 2
        self.current_reward_location = self.reward_locations[reward_index]
        
        # Reward based on reaching goal location
        if self.state == self.current_reward_location:
            # goal_reward = (self.decay ** (self.time_step % (self.time_size//2)))
            goal_reward = 2
        else:
            goal_reward = 0
        
        reward = move_penalty + goal_reward

        time = self.time_step % self.time_size
        # time = (self.time_step // (self.time_size//2)) % 2
        
        return self.state, time, reward

    def render_init(self):
        pygame.init()
        self.cell_size = 100

        # Load sprites
        self.agent_sprite = pygame.image.load('assets/pika.jpg')
        self.agent_sprite = pygame.transform.scale(self.agent_sprite, (self.cell_size, self.cell_size))

        self.goal_sprite = pygame.image.load('assets/rawst.jpg')
        self.goal_sprite = pygame.transform.scale(self.goal_sprite, (self.cell_size, self.cell_size))

        self.top_bar_height = 50
        self.screen_size = self.grid_size * self.cell_size
        self.window_height = self.screen_size + self.top_bar_height
        
        # Update screen creation
        self.screen = pygame.display.set_mode((self.screen_size, self.window_height))
        pygame.display.set_caption('Grid World')

    def render(self, last_action=None, total_reward=0, path='output'):
        # Clear the screen
        self.screen.fill((255, 255, 255))  # White background
        
        font = pygame.font.Font(None, 36)
        
        # Create top info bar background
        pygame.draw.rect(self.screen, (230, 230, 230), (0, 0, self.screen_size, self.top_bar_height))
        
        # Display action taken
        action_names = ['No-op', 'Left', 'Down', 'Right', 'Up']
        if last_action is not None:
            action_text = f"Action: {action_names[last_action]}"
            action_surface = font.render(action_text, True, (0, 0, 0))
            self.screen.blit(action_surface, (10, 10))
        
        # Display total reward
        reward_text = f"Total Reward: {total_reward:.2f}"
        reward_surface = font.render(reward_text, True, (0, 0, 0))
        self.screen.blit(reward_surface, (self.screen_size - 250, 10))
        
        # Draw grid lines
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), 
                             (x, self.top_bar_height), 
                             (x, self.window_height))
        for y in range(self.top_bar_height, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), 
                             (0, y), 
                             (self.screen_size, y))
        
        # Render goal with transparency and border
        if self.goal_sprite:
            goal_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            
            # Draw black border
            border_rect = pygame.Rect(0, 0, self.cell_size, self.cell_size)
            pygame.draw.rect(goal_surface, (0, 0, 0), border_rect, 3)
            
            # Draw slightly smaller transparent goal sprite inside the border
            inner_rect = pygame.Rect(3, 3, self.cell_size-6, self.cell_size-6)
            goal_subsurface = pygame.transform.scale(self.goal_sprite, (self.cell_size-6, self.cell_size-6))
            goal_subsurface.set_alpha(150)  # Transparency
            goal_surface.blit(goal_subsurface, inner_rect)
            
            # Blit the goal surface
            goal_x = self.current_reward_location[0] * self.cell_size
            goal_y = self.current_reward_location[1] * self.cell_size + self.top_bar_height
            self.screen.blit(goal_surface, (goal_x, goal_y))
        
        # Render agent with transparency and border
        if self.agent_sprite:
            agent_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            
            # Draw black border
            border_rect = pygame.Rect(0, 0, self.cell_size, self.cell_size)
            pygame.draw.rect(agent_surface, (0, 0, 0), border_rect, 3)
            
            # Draw slightly smaller transparent agent sprite inside the border
            inner_rect = pygame.Rect(3, 3, self.cell_size-6, self.cell_size-6)
            agent_subsurface = pygame.transform.scale(self.agent_sprite, (self.cell_size-6, self.cell_size-6))
            agent_subsurface.set_alpha(200)  # Transparency
            agent_surface.blit(agent_subsurface, inner_rect)
            
            # Blit the agent surface
            agent_x = self.state[0] * self.cell_size
            agent_y = self.state[1] * self.cell_size + self.top_bar_height
            self.screen.blit(agent_surface, (agent_x, agent_y))
        
        # Update display
        pygame.display.flip()

        # Save display to png
        pygame.image.save(self.screen, path + '/' + f'{self.time_step:03d}.png')
        
        # Control frame rate
        pygame.time.delay(100)

    def close(self):
        pygame.quit()
