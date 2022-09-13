from Bot import Bot
from GameAction import GameAction
from GameState import GameState
from typing import List, Callable
import random
import math
import numpy as np

class LocalSearchBot(Bot):
    # Inisialisasi Variable awal
    def __init__( initial_temperature: float = 0, schedule: Callable[[int], float] = lambda t: math.e ** (-t), precision: float = 1E-300, is_player1: bool = False) -> None:
        initial_temperature = initial_temperature
        schedule = schedule
        precision = precision
        is_player1 = is_player1

    # Pemilihan aksi yang akan dilakukan agent
    def get_action( state: GameState) -> GameAction:
        current = get_random_action(state)
        time = 1
        while True:
            # Perhitungan delta dengan presisi 1e-300
            current_temperature = schedule(time)
            if abs(current_temperature - initial_temperature) <= precision:
                break

            next = get_random_action(state)
            delta = get_value(state, next) - get_value(state, current)

            # Jika delta positif atau tolerable maka ambil langkah selanjutnya
            if delta > 0 or random.random() < math.e ** (delta / current_temperature):
                current = next
            time += 1

        return current

    # Pemilihan aksi random dari list yang tersedia
    def get_random_action( state: GameState) -> GameAction:
        actions = generate_actions(state)
        return random.choice(actions)

    # Generate list aksi yang bisa dilakukan
    def generate_actions( state: GameState) -> List[GameAction]:
        row_positions = generate_positions(state.row_status)
        col_positions = generate_positions(state.col_status)
        actions: List[GameAction] = []

        for position in row_positions:
            actions.append(GameAction("row", position))
        for position in col_positions:
            actions.append(GameAction("col", position))

        return actions

    # Generate posisi dari setiap garis yang masih kosong
    def generate_positions( matrix: np.ndarray) -> List[tuple[int, int]]:
        [ny, nx] = matrix.shape
        positions: List[tuple[int, int]] = []

        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))

        return positions

    # Update board
    def get_result( state: GameState, action: GameAction) -> GameState:
        type = action.action_type
        x, y = action.position

        # Dilakukan copy agar tidak mengubah status game
        new_state = GameState(state.board_status.copy(), state.row_status.copy(), state.col_status.copy(), state.player1_turn)
        player_modifier = -1 if new_state.player1_turn else 1
        is_point_scored = False
        val = 1

        [ny, nx] = new_state.board_status.shape
        
        # Pengecekan apakah akan terbentuk box pada move ini
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

        # Jika bot di player 1 maka kita ingin bot mendapatkan point sebesar mungkin maka langkah ini akan diambil
        # Jika bot berada di player 2 maka kita ingin player 1 tidak mendapatkan poin sehingga kita ingin mengambil aksi dimana player 1 tidak mendapatkan poin
        new_state = new_state._replace(player1_turn = not (new_state.player1_turn ^ is_point_scored))
        return new_state

    # Utility function dengan nilai absolute 1 jika box terbentuk. 
    def get_value( state: GameState, action: GameAction) -> float:
        new_state = get_result(state, action)
        [ny, nx] = new_state.board_status.shape
        utility = 0

        for y in range(ny):
            for x in range(nx):
                if new_state.board_status[y, x] == 4:
                    utility += 1
                elif new_state.board_status[y, x] == -4:
                    utility -= 1

        return -utility if is_player1 else utility