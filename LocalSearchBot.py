from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import random
import numpy as np

class LocalSearchBot(Bot):
    def get_action(self, state: GameState) -> GameAction:
        return super().get_action(state)