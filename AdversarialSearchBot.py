from Bot import Bot
from GameAction import GameAction
from GameState import GameState
from typing import List
import numpy as np

class AdversarialSearchBot(Bot):
    # Inisialisasi Variable awal
    def __init__(self, max_depth: int = 5, is_player1: bool = False):
        self.max_depth = max_depth
        self.is_player1 = is_player1

    # Pemilihan aksi yang dilakukan agent
    def get_action(self, state: GameState) -> GameAction:
        actions = self.generate_actions(state)
        utilities = np.array([self.get_minimax_value(self.get_result(state, action)) for action in actions])
        index = np.random.choice(np.flatnonzero(utilities == utilities.max()))
        return actions[index]

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
    def generate_positions(self, matrix: np.ndarray) -> List[tuple[int, int]]:
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

    def get_minimax_value(self, state: GameState, depth: int = 0, alpha: float = -np.inf, beta: float = np.inf) -> float:
        # Udah ketemu solusi tinggal dihitung dengan fungsi objektif
        # Dilakukan pembatasan depth agar tidak terlalu lama
        if self.terminal_test(state) or depth == self.max_depth:
            return self.get_utility(state)
        
        # Jika belum ketemu, maka akan dicari solusinya dengan dfs dengan turn yang bergantian. 
        # Jika nilai terbaik dari maximizer sudah sama atau melebihi nilai terbaik dari minimzer (alpha lebih dari sama dengan beta)
        # Pencarian neighbor dapat dihentikan karena dapat dipastikan nilai minimum yang kita cari merupakan langkah optimum musuh
        elif not (state.player1_turn ^ self.is_player1):
            value = -np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = max(value, self.get_minimax_value(self.get_result(state, action), depth + 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = np.inf
            actions = self.generate_actions(state)
            for action in actions:
                value = min(value, self.get_minimax_value(self.get_result(state, action), depth + 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # Pengecekan apakah terminal state sudah valid
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

        # TODO: Mungkin bisa dicoba dengan fungsi yang lebih mendeskrispsikan kondisi game
        for y in range(ny):
            for x in range(nx):
                if state.board_status[y, x] == 4:
                    utility += 1
                elif state.board_status[y, x] == -4:
                    utility -= 1

        return -utility if self.is_player1 else utility