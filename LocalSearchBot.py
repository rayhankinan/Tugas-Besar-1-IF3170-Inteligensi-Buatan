from Bot import Bot
from GameAction import GameAction
from GameState import GameState
from typing import List, Callable
import random
import math
import numpy as np
from time import time

TIMEOUT = 4.995

class LocalSearchBot(Bot):
    # Inisialisasi Variable awal
    def __init__(
        self,
        end_temperature: float = 0,
        schedule: Callable[[int], float] = lambda t: math.e ** (-t / 100),
        precision: float = 1e-100,
    ) -> None:
        self.end_temperature = end_temperature
        self.schedule = schedule
        self.precision = precision
        self.is_player1 = True
        self.global_time = 0

    # Pemilihan aksi yang akan dilakukan agent
    def get_action(self, state: GameState) -> GameAction:
        self.is_player1 = state.player1_turn

        current = self.get_random_action(state)
        start_time = 1
        self.global_time = time() + TIMEOUT
        while True:
            # Perhitungan delta dengan presisi 1e-300
            current_temperature = self.schedule(start_time)
            if abs(current_temperature - self.end_temperature) <= self.precision or time() >= self.global_time:
                break

            next = self.get_random_action(state)
            delta = self.get_value(state, next) - \
                self.get_value(state, current)

            # Jika delta positif atau tolerable maka ambil langkah selanjutnya
            if delta > 0 or random.random() < math.e ** (delta / current_temperature):
                current = next
            start_time += 1

        return current

    # Pemilihan aksi random dari list yang tersedia
    def get_random_action(self, state: GameState) -> GameAction:
        actions = self.generate_actions(state)
        return random.choice(actions)

    # Generate list aksi yang bisa dilakukan
    def generate_actions(self, state: GameState) -> List[GameAction]:
        row_positions = self.generate_positions(state.row_status)
        col_positions = self.generate_positions(state.col_status)
        actions: List[GameAction] = []

        for position in row_positions:
            actions.append(GameAction("row", position))
        for position in col_positions:
            actions.append(GameAction("col", position))

        return actions

    # Generate posisi dari setiap garis yang masih kosong
    def generate_positions(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape
        positions: List[tuple[int, int]] = []

        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))

        return positions

    # Update board
    def get_result(self, state: GameState, action: GameAction) -> GameState:
        type = action.action_type
        x, y = action.position

        # Dilakukan copy agar tidak mengubah status game
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

        # Pengecekan apakah akan terbentuk box pada move ini
        if y < ny and x < nx:
            new_state.board_status[y, x] = (
                abs(new_state.board_status[y, x]) + val
            ) * player_modifier
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True

        if type == "row":
            new_state.row_status[y, x] = 1
            if y > 0:
                new_state.board_status[y - 1, x] = (
                    abs(new_state.board_status[y - 1, x]) + val
                ) * player_modifier
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True
        elif type == "col":
            new_state.col_status[y, x] = 1
            if x > 0:
                new_state.board_status[y, x - 1] = (
                    abs(new_state.board_status[y, x - 1]) + val
                ) * player_modifier
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True

        # Jika bot di player 1 maka kita ingin bot mendapatkan point sebesar mungkin maka langkah ini akan diambil
        # Jika bot berada di player 2 maka kita ingin player 1 tidak mendapatkan poin sehingga kita ingin mengambil aksi dimana player 1 tidak mendapatkan poin
        new_state = new_state._replace(
            player1_turn=not (new_state.player1_turn ^ is_point_scored)
        )
        return new_state

    # Utility function dengan nilai absolute 1 jika box terbentuk.
    def get_value(self, state: GameState, action: GameAction) -> float:

        new_state = self.get_result(state, action)
        [ny, nx] = new_state.board_status.shape
        utility = 0

        # Menghitung jumlah box yang terbentuk
        box_won = 0
        box_lost = 0
        for y in range(ny):
            for x in range(nx):
                if self.is_player1:
                    if new_state.board_status[y, x] == -4:
                        utility += 1
                        box_won += 1
                    elif new_state.board_status[y, x] == 4 or abs(new_state.board_status[y, x]) == 3:
                        utility -= 1
                        box_lost += 1
                else:
                    if new_state.board_status[y, x] == -4 or abs(new_state.board_status[y, x]) == 3:
                        utility -= 1
                        box_lost += 1
                    elif new_state.board_status[y, x] == 4:
                        utility += 1
                        box_won += 1

        # Chain rule
        if self.chain_count(new_state) % 2 == 0 and self.is_player1:
            utility += 1
        elif self.chain_count(new_state) % 2 != 0 and not self.is_player1:
            utility += 1

        # Win/Lose Heuristics
        if box_won >= 5:
            utility = np.inf
        elif box_lost >= 5:
            utility = -np.inf

        return utility

    # Count the number of long chain(s)
    def chain_count(self, state: GameState) -> int:

        chain_count = 0
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
                chain_count += 1

        return chain_count

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