from Bot import Bot
from GameAction import GameAction
from GameState import GameState
from typing import List, Callable
import random
import math
import numpy as np

class LocalSearchBot(Bot):
    def __init__(self, initial_temperature: float = 0, schedule: Callable[[int], float] = lambda t: math.e ** (-t), precision: float = 1E-18, is_player1: bool = False) -> None:
        self.initial_temperature = initial_temperature
        self.schedule = schedule
        self.precision = precision
        self.is_player1 = is_player1

    def get_action(self, state: GameState) -> GameAction:
        current = self.get_random_action(state)
        time = 1
        while True:
            current_temperature = self.schedule(time)
            if abs(current_temperature - self.initial_temperature) <= self.precision:
                break

            next = self.get_random_action(state)
            delta = self.get_value(state, next) - self.get_value(state, current)

            print(current, next, delta) # DELETE THIS LINE

            if delta > 0 or random.random() < math.e ** (delta / current_temperature):
                current = next
            time += 1

        print("Selected", current) # DELETE THIS LINE
        return current

    def get_random_action(self, state: GameState) -> GameAction:
        actions = self.generate_actions(state)
        return random.choice(actions)

    def generate_actions(self, state: GameState) -> List[GameAction]:
        row_positions = self.generate_positions(state.row_status)
        col_positions = self.generate_positions(state.col_status)
        actions: List[GameAction] = []

        for position in row_positions:
            actions.append(GameAction("row", position))
        for position in col_positions:
            actions.append(GameAction("col", position))

        return actions

    def generate_positions(self, matrix: np.ndarray) -> List[tuple[int, int]]:
        [ny, nx] = matrix.shape
        positions: List[tuple[int, int]] = []

        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))

        return positions

    def get_result(self, state: GameState, action: GameAction) -> GameState:
        type = action.action_type
        x, y = action.position

        new_state = GameState(state.board_status.copy(), state.row_status.copy(), state.col_status.copy(), state.player1_turn)
        player_modifier = -1 if new_state.player1_turn else 1
        is_point_scored = False
        val = 1

        [ny, nx] = new_state.board_status.shape
        if y < ny and x < nx:
            new_state.board_status[y, x] = (abs(new_state.board_status[y, x]) + val) * player_modifier
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True

        if type == "row":
            new_state.row_status[y, x] = 1
            if y > 0:
                new_state.board_status[y - 1, x] = (abs(new_state.board_status[y - 1, x]) + val) * player_modifier
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True
        elif type == "col":
            new_state.col_status[y, x] = 1
            if x > 0:
                new_state.board_status[y, x - 1] = (abs(new_state.board_status[y, x - 1]) + val) * player_modifier
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True

        new_state = new_state._replace(player1_turn = not (new_state.player1_turn ^ is_point_scored))
        return new_state

    def get_value(self, state: GameState, action: GameAction) -> float:
        new_state = self.get_result(state, action)
        [ny, nx] = new_state.board_status.shape
        utility = 0

        for y in range(ny):
            for x in range(nx):
                if new_state.board_status[y, x] == 4:
                    utility += 1
                elif new_state.board_status[y, x] == -4:
                    utility -= 1

        return -utility if self.is_player1 else utility