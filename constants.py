from enum import Enum

BAD, DEFECT, MEAN = 0, 0, 0
GOOD, COOPERATE, NICE = 1, 1, 1


class Strategy(Enum):
    ALWAYS_COOPERATE = (1, 1)
    DISCRIMINATE = (0, 1)
    PARADOXICALLY_DISC = (1, 0)
    ALWAYS_DEFECT = (0, 0)

    def strategy_name(self):
        return self.name.replace("_", " ").title()


class EmotionProfile(Enum):
    COMPETITIVE = 0
    COOPERATIVE = 1

    def ep_name(self):
        return self.name.title()

    def mutate(self):
        # Since there are only two, flip
        return EmotionProfile(1 - self.value)
