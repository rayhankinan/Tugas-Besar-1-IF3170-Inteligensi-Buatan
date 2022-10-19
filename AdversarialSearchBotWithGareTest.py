from AdversarialSearchBot import AdversarialSearchBot
from GameState import GameState
import numpy as np


class AdversarialSearchBotWithGareTest(AdversarialSearchBot):
    def get_utility(self, state: GameState) -> float:
        [ny, nx] = state.board_status.shape
        utility = 0

        # == Gare test
        for y in range(ny):
            for x in range(nx):
                if self.is_player1 == state.player1_turn:
                    if abs(state.board_status[y, x]) == 4:
                        utility += 1
                else:
                    if abs(state.board_status[y, x]) == 4:
                        utility -= 1

        if self.terminal_test(state):
            if self.is_player1 == state.player1_turn:
                utility = np.inf
            else:
                utility = -np.inf

        return utility
