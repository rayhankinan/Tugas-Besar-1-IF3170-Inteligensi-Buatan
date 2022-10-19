from AdversarialSearchBot import AdversarialSearchBot
from GameState import GameState
from typing import List


class AdversarialSearchBotWithChaining(AdversarialSearchBot):
    def get_utility(self, state: GameState) -> float:
        utility = super().get_utility(state)
        chain_count = self.chain_count(state)

        # == Chain rule
        if self.is_player1:
            if chain_count % 2 == 0:
                utility += 3
        else:
            if chain_count % 2 != 0:
                utility += 3

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

    # == Find adjacent box(es) which can build chain
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
