from time import time
from AdversarialSearchBot import AdversarialSearchBot
from GameState import GameState
from GameAction import GameAction
import numpy as np
from typing import List

TIMEOUT = 5
MAX_DEPTH = 6


class AdversarialSearchBotWithGareTest(AdversarialSearchBot):
    def get_action(self, state: GameState) -> GameAction:
        self.is_player1 = state.player1_turn
        selected_action = None
        self.global_time = time() + TIMEOUT
        for depth in range(MAX_DEPTH):
            if time() >= self.global_time:
                break

            generated_actions = self.generate_actions(state)
            utils = np.array([
                self.get_minimax_value(state=self.get_result(state, action), max_depth=depth+1) for action in generated_actions
            ])

            selected_index = np.random.choice(
                np.flatnonzero(utils == utils.max())
            )
            print(depth + 1, ": ", utils, "SELECTED IDX: ", selected_index)
            selected_action = generated_actions[selected_index]

        return selected_action

    def generate_actions(self, state: GameState) -> List[GameAction]:
        row_positions = self.generate_positions(state.row_status)
        col_positions = self.generate_positions(state.col_status)
        actions: List[GameAction] = []

        # == Add row first to increase heuristics
        for position in row_positions:
            actions.append(GameAction("row", position))
        for position in col_positions:
            actions.append(GameAction("col", position))

        return actions

    def get_result(self, state: GameState, action: GameAction) -> GameState:
        type = action.action_type
        x, y = action.position

        new_state = GameState(
            state.board_status.copy(),
            state.row_status.copy(),
            state.col_status.copy(),
            state.player1_turn,
        )
        # == Bot kan player 2
        # == nah berarti ini player1Turennya pas awal true
        player_modifier = -1 if new_state.player1_turn else 1

        is_point_scored = False
        val = 1

        [ny, nx] = new_state.board_status.shape

        # == Check if this move will make a box
        if y < ny and x < nx:
            new_state.board_status[y, x] = (
                abs(new_state.board_status[y, x]) + val
            ) * player_modifier
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True

        # == modified and check for row statuses
        if type == "row":
            new_state.row_status[y, x] = 1
            if y > 0:
                new_state.board_status[y - 1, x] = (
                    abs(new_state.board_status[y - 1, x]) + val
                ) * player_modifier
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True

        # == modified and check for col statuses
        elif type == "col":
            new_state.col_status[y, x] = 1
            if x > 0:
                new_state.board_status[y, x - 1] = (
                    abs(new_state.board_status[y, x - 1]) + val
                ) * player_modifier
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True
        is_player1_turn = not new_state.player1_turn if not is_point_scored else new_state.player1_turn
        new_state = new_state._replace(
            player1_turn=is_player1_turn
        )
        return new_state

    def get_minimax_value(
        self,
        state: GameState,
        depth: int = 0,
        max_depth: int = 0,
        alpha: float = -np.inf,
        beta: float = np.inf,
    ) -> float:

        # == Checking the value of terminal state
        if (
            self.terminal_test(state)
            or depth == max_depth
            or time() >= self.global_time
        ):
            return self.get_utility(state)

        # == First is maximizer then minimizer if our turn
        if self.is_player1 == state.player1_turn:
            value = -np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = max(
                    value,
                    self.get_minimax_value(
                        self.get_result(state, action),
                        depth=depth + 1,
                        max_depth=max_depth,
                        alpha=alpha,
                        beta=beta
                    ),
                )
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = min(
                    value,
                    self.get_minimax_value(
                        self.get_result(state, action),
                        depth=depth + 1,
                        max_depth=max_depth,
                        alpha=alpha,
                        beta=beta
                    ),
                )
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def get_utility(self, state: GameState) -> float:
        [ny, nx] = state.board_status.shape
        utility = 0

        bot_turn = self.is_player1 == state.player1_turn
        mult = 1 if bot_turn else -1
        # Check
        # for y in range(ny):
        #     for x in range(nx):
        #         board_status = state.board_status[y, x]
        #         sign = -1 if board_status < 0 else 1
        # if bot is player 1
        # if board_status is < 0 then it currently mark by bot
        # 4 is priority then 3 then 1 then 2
        # if self.is_player1:
        #     if abs(board_status) == 4:
        #         utility += 50 if sign < 0 else -50
        #     elif abs(board_status) == 3:
        #         utility += 50 if sign < 0 else -50
        #     elif abs(board_status) == 2:
        #         utility += -25 if sign < 0 else 5
        #     else:
        #         utility += 10 if sign < 0 else -10
        # else:
        #     if abs(board_status) == 4:
        #         utility += -50 if sign < 0 else 50
        #     elif abs(board_status) == 3:
        #         utility += -50 if sign < 0 else 50
        #     elif abs(board_status) == 2:
        #         utility += 25 if sign < 0 else -5
        #     else:
        #         utility += -10 if sign < 0 else 10
        for y in range(ny):
            for x in range(nx):
                if self.is_player1:
                    if state.board_status[y, x] == -4:
                        utility += 10
                    elif state.board_status[y, x] == 4:
                        utility -= 2
                else:
                    if state.board_status[y, x] == -4:
                        utility -= 2
                    elif state.board_status[y, x] == 4:
                        utility += 10

        return utility
