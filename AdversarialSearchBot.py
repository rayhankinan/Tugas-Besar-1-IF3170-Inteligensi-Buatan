from time import perf_counter
from Bot import Bot
from GameAction import GameAction
from GameState import GameState
from typing import List
import numpy as np


class AdversarialSearchBot(Bot):

    # Initialize bot
    def __init__(self, max_depth: int = 5, is_player1: bool = False):
        self.max_depth = max_depth
        self.is_player1 = is_player1
        self.is_global_time_stop = False
        self.global_time = 0

    # == Implement get action from bot class
    def get_action(self, state: GameState) -> GameAction:
        actions = self.generate_actions(state)
        utilities = np.array(
            [
                self.get_minimax_value(state=self.get_result(state, action))
                for action in actions
            ]
        )

        if self.is_player1:
            index = np.random.choice(
                np.flatnonzero(utilities == utilities.max()))
            return actions[index]
        else:
            index = np.random.choice(
                np.flatnonzero(utilities == utilities.min()))
            return actions[index]

    # == Generate list of game action
    def generate_actions(self, state: GameState) -> List[GameAction]:
        row_positions = self.generate_positions(state.row_status)
        col_positions = self.generate_positions(state.col_status)
        actions: List[GameAction] = []

        # TODO: Menambahkan heuristik move ordering
        for position in row_positions:
            actions.append(GameAction("row", position))
        for position in col_positions:
            actions.append(GameAction("col", position))

        return actions

    # == Generate valid position
    def generate_positions(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape
        positions: List[tuple[int, int]] = []

        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))

        return positions

    # == Update board
    def get_result(self, state: GameState, action: GameAction) -> GameState:
        type = action.action_type
        x, y = action.position

        # == FIXME
        new_state = GameState(
            state.board_status.copy(),
            state.row_status.copy(),
            state.col_status.copy(),
            state.player1_turn,
        )

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

        new_state = new_state._replace(
            player1_turn=not (new_state.player1_turn ^ is_point_scored)
        )
        return new_state

    def get_minimax_value(
        self,
        state: GameState,
        depth: int = 0,
        alpha: float = -np.inf,
        beta: float = np.inf,
        is_root: bool = True,
        current_time=perf_counter(),
    ) -> float:
        # if is_root:
        #     self.global_time = perf_counter()

        # time_difference = abs(self.global_time - current_time)

        # if time_difference >= 4.8:
        #     self.is_global_time_stop = True

        if (
            self.terminal_test(state)
            or depth == self.max_depth
            # or self.is_global_time_stop
        ):
            # self.global_time = 0
            # self.is_global_time_stop = False
            return self.get_utility(state)

        # Jika belum ketemu, maka akan dicari solusinya dengan dfs dengan turn yang bergantian.
        # Jika nilai terbaik dari maximizer sudah sama atau melebihi nilai terbaik dari minimzer (alpha lebih dari sama dengan beta)
        # Pencarian neighbor dapat dihentikan karena dapat dipastikan nilai minimum yang kita cari merupakan langkah optimum musuh
        if not (state.player1_turn ^ self.is_player1):
            value = -np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = max(
                    value,
                    self.get_minimax_value(
                        self.get_result(state, action),
                        depth + 1,
                        alpha,
                        beta,
                        False,
                        current_time=perf_counter(),
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
                        depth + 1,
                        alpha,
                        beta,
                        False,
                        current_time=perf_counter(),
                    ),
                )
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # == Check if terminal leaf has box
    def terminal_test(self, state: GameState) -> bool:
        [ny, nx] = state.board_status.shape

        for y in range(ny):
            for x in range(nx):
                if abs(state.board_status[y, x]) != 4:
                    return False
        return True

    # Utility function dengan nilai absolute 1 jika box terbentuk.
    def get_utility(self, state: GameState) -> float:
        [ny, nx] = state.board_status.shape
        utility = 0

        # TODO: Menambahkan heuristik transposition table (untuk melakukan caching nilai utility) dengan corner symmetry

        # == Count boxes
        for y in range(ny):
            for x in range(nx):
                if self.is_player1:
                    if state.board_status[y, x] == -4:
                        utility += 1
                    elif state.board_status[y, x] == 4:
                        utility -= 1
                else:
                    if state.board_status[y, x] == -4:
                        utility -= 1
                    elif state.board_status[y, x] == 4:
                        utility += 1

        # == Chain rule
        # if self.chain_count(state) % 2 == 0 and self.is_player1:
        #     utility += 3
        # elif self.chain_count(state) % 2 != 0 and not self.is_player1:
        #     utility += 3

        return utility

    # == Count the number of long chain(s)
    def chain_count(self, state: GameState) -> int:

        num_of_chain = 0
        chain_list: List[List[int]] = []

        for box_num in range(9):

            # Check if box is already part of a chain
            flag = False
            for chain in chain_list:
                if box_num in chain:
                    flag = True
                    break

            if not flag:
                chain_list.append([box_num])
                self.add_chain(state, chain_list, box_num)

        for chain in chain_list:
            if len(chain) >= 3:
                num_of_chain += 1

        return num_of_chain

    # Find adjacent box(es) which can build chain
    def add_chain(self, state: GameState, chain_list: List[List[int]], box_num):

        neighbors_num = [box_num - 1, box_num - 3, box_num + 1, box_num + 3]

        for idx in range(len(neighbors_num)):
            if (
                neighbors_num[idx] < 0
                or neighbors_num[idx] > 8
                or (idx % 2 == 0 and neighbors_num[idx] // 3 != box_num // 3)
            ):
                continue

            flag = False
            for chain in chain_list:
                if neighbors_num[idx] in chain:
                    flag = True
                    break

            if not flag and idx % 2 == 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.col_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain(state, chain_list, neighbors_num[idx])

            if not flag and idx % 2 != 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.row_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain(state, chain_list, neighbors_num[idx])
