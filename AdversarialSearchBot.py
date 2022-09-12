from Bot import Bot
from GameAction import GameAction
from GameState import GameState
from typing import List
import numpy as np

class AdversarialSearchBot(Bot):
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def get_action(self, state: GameState) -> GameAction:
        actions = self.generate_actions(state)
        utilities = np.array([self.get_minimax_value(self.get_result(state, action)) for action in actions])
        index = np.random.choice(np.flatnonzero(utilities == utilities.max()))
        print(utilities) # DELETE THIS LATER
        print(index) # DELETE THIS LATER
        return actions[index]

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
        player_modifier = 1 if new_state.player1_turn else -1
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
        # print(action, new_state) # DELETE THIS LATER
        return new_state

    def get_minimax_value(self, state: GameState, depth: int = 0, alpha: float = -np.inf, beta: float = np.inf) -> float:
        if self.terminal_test(state) or depth == self.max_depth:
            # print(self.get_utility(state)) # DELETE THIS LATER
            return self.get_utility(state)
        elif not state.player1_turn:
            value = -np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = max(value, self.get_minimax_value(self.get_result(state, action), depth + 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            # print(value) # DELETE THIS LATER
            return value
        else:
            value = np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = min(value, self.get_minimax_value(self.get_result(state, action), depth + 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            # print(value) # DELETE THIS LATER
            return value

    def terminal_test(self, state: GameState) -> bool:
        [ny, nx] = state.board_status.shape

        for y in range(ny):
            for x in range(nx):
                if abs(state.board_status[y, x]) != 4:
                    return False
        return True

    def get_utility(self, state: GameState) -> float:
        [ny, nx] = state.board_status.shape
        utility = 0

        for y in range(ny):
            for x in range(nx):
                if state.board_status[y, x] == 4:
                    utility += 1
                elif state.board_status[y, x] == -4:
                    utility -= 1

        return -utility