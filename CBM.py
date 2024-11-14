from CBM_Agent import *

class CMB:
    def __init__(self, num_agents):
        super().__init__()
        self.Agents = []
        for i in range(num_agents):
            self.Agents.append(CBMPopulationAgent(20, 0.5, 1, 5, 0.5, 100, 5))