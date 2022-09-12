from Bot import Bot
from GameAction import GameAction
from GameState import GameState
from typing import List
import numpy as np

class AdversarialSearchBot(Bot):
    def get_action(self, state: GameState) -> GameAction:
        actions = self.generate_actions(state)
        print([self.get_minimax_value(self.get_result(state, action)) for action in actions]) # DELETE THIS LATER
        index = np.argmax([self.get_minimax_value(self.get_result(state, action)) for action in actions])
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

    def get_result(self, state: GameState, action: GameAction) -> GameState: # BUG: CHECK WHETHER BOX IS FULL OR NOT
        type = action.action_type
        x, y = action.position

        new_state = GameState(state.board_status.copy(), state.row_status.copy(), state.col_status.copy(), state.player1_turn)
        player_modifier = 1 if new_state.player1_turn else -1
        is_point_scored = False
        val = 1

        [ny, nx] = new_state.board_status.shape
        if y < ny and x < nx:
            new_state.board_status[y, x] = abs(new_state.board_status[y, x] + val) * player_modifier
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True

        if type == "row":
            new_state.row_status[y, x] = 1
            if y >= 1:
                new_state.board_status[y - 1, x] = (abs(new_state.board_status[y - 1, x]) + val) * player_modifier
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True
        elif type == "col":
            new_state.col_status[y, x] = 1
            if x >= 1:
                new_state.board_status[y, x - 1] = (abs(new_state.board_status[y, x - 1]) + val) * player_modifier
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True

        new_state = new_state._replace(player1_turn = not (new_state.player1_turn ^ is_point_scored))
        # print(action, new_state) # DELETE THIS LATER
        return new_state

    def get_minimax_value(self, state: GameState, alpha: float = -np.inf, beta: float = np.inf) -> float:
        if self.terminal_test(state):
            return self.get_utility(state)
        elif not state.player1_turn:
            value = -np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = max(value, self.get_minimax_value(self.get_result(state, action), alpha, beta)) # BUG HERE
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = min(value, self.get_minimax_value(self.get_result(state, action), alpha, beta)) # BUG HERE
                beta = min(beta, value)
                if beta <= alpha:
                    break
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
                if state.board_status[y, x] > 0:
                    utility -= 1
                else:
                    utility += 1

        return utility