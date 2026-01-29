from enum import Enum

BAD, DEFECT, MEAN, REGRET = 0, 0, 0, 0
GOOD, COOPERATE, NICE, JOY = 1, 1, 1, 1


class Strategy(Enum):
    ALWAYS_COOPERATE = (1, 1)
    DISCRIMINATE = (0, 1)
    PARADOXICALLY_DISC = (1, 0)
    ALWAYS_DEFECT = (0, 0)

    def strategy_name(self):
        return self.name.replace("_", " ").title()


class EmotionExpression(Enum):
    NEUTRAL = 0
    ANGER = 1
    JOY = 2
    REGRET = 3

    def __str__(self):
        return self.name.title()


class EmotionProfile(Enum):

    COMPETITIVE = ((EmotionExpression.NEUTRAL, EmotionExpression.JOY),
                   (EmotionExpression.ANGER, EmotionExpression.REGRET))
    COOPERATIVE = ((EmotionExpression.NEUTRAL, EmotionExpression.REGRET),
                   (EmotionExpression.ANGER, EmotionExpression.JOY))

    def ep_name(self):
        return self.name.title()

    def mutate(self):
        # Since there are only two, flip
        if self == EmotionProfile.COMPETITIVE:
            return EmotionProfile.COOPERATIVE
        else:
            return EmotionProfile.COMPETITIVE





