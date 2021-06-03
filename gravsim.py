import math
from math import cos, sin, sqrt
from multiprocessing import Process, Lock
from os import getcwd
from tkinter import Tk, filedialog
import json


import keyboard

import pygame
import numpy as np
import pygame.surfarray as surfarray



pygame.init()


class AdvancedClock():
    def __init__(self, fps=None):
        self._clock = pygame.time.Clock()
        self._ms_since_tick = 0
        if fps is not None:
            self.fps = fps
    
    def tick(self):
        ms_since_tick = self._ms_since_tick
        self._ms_since_tick = 0
        return ms_since_tick + self._clock.tick()

    def since_tick(self):
        self._ms_since_tick += self._clock.tick()
        return self._ms_since_tick


screen = None

def get_json_filepath():
    m = Tk()
    filepath = filedialog.askopenfilename(
        initialdir=getcwd() + '\\',
        title='Select a Telegram Export JSON File',
        filetypes=(('JSON files', '*.json*'),('All files', '*.*')),
        master=m
    )
    m.destroy()
    return filepath

def init_window(array):
    global screen
    screen = pygame.display.set_mode(array.shape[:2])
    update_window(array)


def update_window(array):
    surfarray.blit_array(screen, array)
    pygame.display.update()


canvas = np.zeros((1000, 700, 3), dtype=np.uint8)

colors = {
    'cyan': np.array([0, 255, 255], dtype=np.uint8),
    'blue': np.array([0, 0, 255], dtype=np.uint8),
    'orange': np.array([255, 127, 0], dtype=np.uint8),
    'yellow': np.array([255, 255, 0], dtype=np.uint8),
    'green': np.array([0, 255, 0], dtype=np.uint8),
    'purple': np.array([255, 0, 255], dtype=np.uint8),
    'red': np.array([255, 0, 0], dtype=np.uint8),
    'black': np.array([0, 0, 0], dtype=np.uint8),
    'white': np.array([255, 255, 255], dtype=np.uint8),
}

background_color = colors['black']

def add_with_factor(a1, a2, factor):
    return [a1[0] + a2[0] * factor, a1[1] + a2[1] * factor]

class Planet():
    def __init__(self,
            mass,
            pos=(0, 0),
            velo=(0, 0),
            acc=(0, 0),
            ):
        self.mass = mass
        self.pos = list(pos)
        self.velocity = list(velo)
        self.acceleration = list(acc)

    def advance_1t(self):
        self.velocity = add_with_factor(self.velocity, self.acceleration, spt)
        self.pos = add_with_factor(self.pos, self.velocity, spt)

    def apply_abs_force(self, force, direction):
        self.acceleration[0] += force * sin(direction) / self.mass
        self.acceleration[1] += force * cos(direction) / self.mass

    def reset_acceleration(self):
        self.acceleration = [0, 0]

    def get_direction_to(self, planet):
        return math.atan2((planet.pos[0] - self.pos[0]), (planet.pos[1] - self.pos[1]))

    def draw():
        pass



class CirclePlanet(Planet):
    def __init__(self,
            radius,
            mass,
            color=None,
            pos=(0, 0),
            velo=(0, 0),
            acc=(0, 0)
            ):
        super().__init__(mass, pos, velo, acc)
        self.radius = radius
        self.pixels = np.zeros((2*radius + 1, 2*radius + 1), dtype=bool)
        diag_len = int(sqrt(2)*0.5*radius)
        start = radius-diag_len
        end = radius+diag_len
        self.pixels[start:end+1, start:end+1] = True
        for x in range(start, end+1):
            for y in range(0, start):
                self.pixels[x, y] = math.hypot(radius-x, radius-y) <= radius
        quarter_section = self.pixels[start:end+1, :start]
        self.pixels[end+1:, start:end+1] = np.rot90(quarter_section, k=1)
        self.pixels[start:end+1, end+1:] = np.rot90(quarter_section, k=2)
        self.pixels[:start, start:end+1] = np.rot90(quarter_section, k=3)
        if isinstance(color, list):
            self.color = np.array(color, dtype=np.uint8)
        elif isinstance(color, str):
            if color in colors:
                self.color = colors[color]
            else:
                if len(color) == 7 and color[0] == '#':
                    arr = [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                    if all([0 <= elem < 256 for elem in arr]):
                        self.color = np.array(arr, dtype=np.uint8)
        self.color = colors['white'] if not hasattr(self, 'color') else self.color

    def draw(self, color=None):
        pos = [self.pos[0] / scale, self.pos[1] / scale]
        canvas[
            max(0, int(pos[0] - self.radius + 0.5)):\
            min(canvas.shape[0], int(pos[0] + self.radius + 1.5)),
            max(0, int(pos[1] - self.radius + 0.5)):\
            min(canvas.shape[1], int(pos[1] + self.radius + 1.5)),
            :]\
            [
                self.pixels[\
                    max(0, int(self.radius - pos[0] + 0.5)):\
                    max(min(self.pixels.shape[0],
                        int(self.radius + canvas.shape[0] - pos[0] + 0.5)),
                        0),
                    max(0, int(self.radius - pos[1] + 0.5)):\
                    max(min(self.pixels.shape[1],
                        int(self.radius + canvas.shape[1] - pos[1] + 0.5)),
                        0)
                ]
            ] = self.color if color is None else color

'''
def check(radius):
    c = Circle(radius, 1)
    count = 0
    for i in range((radius*2)+1):
        for j in range((radius*2)+1):
            if c.pixels[i, j] != ((radius-i)**2 + (radius-j)**2 < radius**2):
                print(radius, i, j)
                count += 1
    return count
'''


planet_list = []
G = 6.67430 * 10**-11
spt = 1000
tps = 10
fps = 120
scale = 2*10**6

def dist(planet_a, planet_b):
    return math.hypot(*[a - b for a, b in zip(planet_a.pos, planet_b.pos)])

def advance_1t(draw=True):
    for planet in planet_list:
        planet.reset_acceleration()
        if draw: planet.draw(background_color)
    
    for i in range(len(planet_list)):
        for j in range(i):
            planet_a = planet_list[i]
            planet_b = planet_list[j]
            F = G * planet_a.mass * planet_b.mass / dist(planet_a, planet_b)**2
            planet_a.apply_abs_force(F, planet_a.get_direction_to(planet_b))
            planet_b.apply_abs_force(F, planet_b.get_direction_to(planet_a))

    for planet in planet_list:
        planet.advance_1t()
        if draw: planet.draw()


if __name__ == '__main__':
    init_window(canvas)
    
    tick_clock = AdvancedClock()
    frame_clock = AdvancedClock()
    event_clock = AdvancedClock()

    json_dict = json.load(open(get_json_filepath(), 'r', encoding='UTF-8'))

    if json_dict.get('constants') is not None:
        consts = json_dict.get('constants')
        G = G if consts.get('G') is None else consts['G']
        spt = spt if consts.get('spt') is None else consts['spt']
        tps = tps if consts.get('tps') is None else consts['tps']
        fps = fps if consts.get('fps') is None else consts['fps']
        scale = scale if consts.get('scale') is None else consts['scale']
    if json_dict.get('planets') is not None:
        for planet in json_dict.get('planets'):
            if isinstance(json_dict['planets'], dict):
                planet_list.append(CirclePlanet(**json_dict['planets'][planet]))
            elif isinstance(json_dict['planets'], list):
                planet_list.append(CirclePlanet(**planet))
    
##    planet_list.append(
##        CirclePlanet(
##            20,
##            5.972*10**24,
##            pos=(500*10**6, 350*10**6),
##            velo=(0, -12.77),
##            color=colors['green']
##            )
##        )
##    planet_list.append(
##        CirclePlanet(
##            10,
##            7.348*10**22,
##            pos=(115*10**6, 350*10**6),
##            velo=(0, 1027.778),
##            color=[127, 127, 127]
##            )
##        )
    
    def advance_update(draw=True):
        advance_1t(draw)
        update_window(canvas)

    
    #keyboard.add_hotkey('a', advance_update)
    
    paused = False
    end = False
    
    def run():
        global end, paused
        while not end:
            if event_clock.since_tick() > 5:
                event_clock.tick()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_ESCAPE:
                            end = True
            if tick_clock.since_tick() > (1 / tps) and not paused:
                tick_clock.tick()
                advance_1t()
            if frame_clock.since_tick() > (1 / fps) and not paused:
                frame_clock.tick()
                update_window(canvas)
    
    run()