import heapq
import sys
from itertools import *
import itertools
from constants import *
from environment import *
from state import State

"""
solution.py

This file is a template you should use to implement your solution.

You should implement 

COMP3702 2024 Assignment 1 Support Code
"""


class Solver:

    def __init__(self, environment, loop_counter):
        self.environment = environment
        self.loop_counter = loop_counter
        self.counter = itertools.count()
        self.target_cubes = [self.axial_to_cube(pos) for pos in self.environment.target_list]


    # === Uniform Cost Search ==========================================================================================
    def solve_ucs(self):
        """
        Find a path which solves the environment using Uniform Cost Search (UCS).
        :return: path (list of actions, where each action is an element of BEE_ACTIONS)
        """

        # initialize the starting state
        start_state = self.environment.get_init_state()

        # initialize the frontier with the starting state and an empty path
        frontier = []
        heapq.heappush(frontier, (0, next(self.counter), start_state, []))  # (cost, counter, state, path)

        # track visited states with their respective costs
        visited = {start_state: 0}  # Initialize visited with the initial state and its cost

        # main ucs loop
        while frontier:
            self.loop_counter.inc()  # increment loop counter

            # get the state with the lowest cost from the frontier
            cost, _, state, path = heapq.heappop(frontier)

            # if the current state is the goal state, return the path
            if self.environment.is_solved(state):
                return path

            # If this state was visited with a lower cost, skip it
            if cost > visited[state]:
                continue

            # explore all possible actions from the current state
            for action in BEE_ACTIONS:
                success, action_cost, new_state = self.environment.perform_action(state, action)
                if success:
                    new_cost = cost + action_cost
                    # add the new state to the frontier if it's not visited or the new path is cheaper
                    if new_state not in visited or new_cost < visited[new_state]:
                        visited[new_state] = new_cost
                        new_path = path + [action]
                        heapq.heappush(frontier, (new_cost, next(self.counter), new_state, new_path))

        return None  # no solution found

    # === A* Search ====================================================================================================

    def preprocess_heuristic(self):
        """
        Perform pre-processing (e.g. pre-computing repeatedly used values) necessary for your heuristic,
        """

        pass


    def compute_heuristic(self, state):
        """
        Compute a heuristic value h(n) for the given state.
        :param state: given state (GameState object)
        :return a real number h(n)
        """

        # convert beebot's position from axial coordinates to cube coordinates
        bee_cube = self.axial_to_cube(state.BEE_posit)
        min_distance = float('inf')

        # iterate over all widgets and find the closest widget that is not yet placed
        for i, widget_center in enumerate(state.widget_centres):
            # use widget_get_occupied_cells method to get the current position occupied by widget
            widget_cells = widget_get_occupied_cells(self.environment.widget_types[i], widget_center,
                                                     state.widget_orients[i])

            # check if the widget is already at the target position
            if not any(cell in self.environment.target_list for cell in widget_cells):
                # if the widget is not at the target position, calculate hte distance from beebot to the widget center
                widget_cube = self.axial_to_cube(widget_center)
                distance = self.cube_distance(bee_cube, widget_cube)
                min_distance = min(min_distance, distance)

        # if all widgets are already placed at target positions, return 0
        if min_distance == float('inf'):
            return 0

        return min_distance


    def axial_to_cube(self, axial):
        """
        Convert axial coordinates to cube coordinates.
        :param axial: (row, col) tuple
        :return: (x, y, z) tuple
        """
        row, col = axial
        x = col
        z = row - (col - (col & 1)) // 2
        y = -x - z
        return (x, y, z)

    def cube_distance(self, cube1, cube2):
        """
        Calculate the distance between two cube coordinates.
        :param cube1: (x1, y1, z1) tuple
        :param cube2: (x2, y2, z2) tuple
        :return: Cube distance
        """
        return max(abs(cube1[0] - cube2[0]), abs(cube1[1] - cube2[1]), abs(cube1[2] - cube2[2]))

    def solve_a_star(self):
        """
        Find a path which solves the environment using A* search.
        :return: path (list of actions, where each action is an element of BEE_ACTIONS)
        """

        # initialize the starting state
        start_state = self.environment.get_init_state()

        #initialize the frontier with the starting state and its heuristic value
        frontier = []
        start_heuristic = self.compute_heuristic(start_state)
        heapq.heappush(frontier,
                       (start_heuristic, next(self.counter), start_state, []))  # (heuristic + cost, counter, state, path)

        # track visited states with their respective g_cost
        visited = {start_state: 0}

        # main A* loop
        while frontier:
            self.loop_counter.inc()  # increment loop counter

            # get the state with the lowest estimated cost (f = g + h) from the frontier
            heuristic_cost, _, state, path = heapq.heappop(frontier)

            # calculate the g_cost (actual cost to reach the state)
            g_cost = heuristic_cost - self.compute_heuristic(state)

            # check if the current state is the goal state
            if self.environment.is_solved(state):
                return path

            # if this state was visited with a lower cost, skip it
            if g_cost > visited[state]:
                continue

            # explore all possible actions from the current state
            for action in BEE_ACTIONS:
                success, action_cost, new_state = self.environment.perform_action(state, action)
                if success:
                    new_g_cost = g_cost + action_cost
                    # add the new state to the frontier if it's not visited or the new path is cheaper
                    if new_state not in visited or new_g_cost < visited[new_state]:
                        visited[new_state] = new_g_cost
                        new_path = path + [action]
                        heapq.heappush(frontier, (
                        new_g_cost + self.compute_heuristic(new_state), next(self.counter), new_state, new_path))

        return None  # no solution found

