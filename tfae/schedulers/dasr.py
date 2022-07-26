from .base import BaseScheduler


def linear(x: float, a: float, b: float, A: float, B: float) -> float:
    """
    Projects x from [a, b] to [A, B]
    """
    return x * (B - A) / (b - a) + (A * b - B * a) / (b - a)


class DASRScheduler(BaseScheduler):
    """ """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        delay: int = 0,
        attack: int = 0,
        sustain: int = 0,
        release: int = 0,
        cycles: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.start_value = start_value
        self.end_value = end_value

        self.delay = delay
        self.attack = attack
        self.sustain = sustain
        self.release = release
        self.cycles = cycles

    @property
    def cycle_duration(self) -> int:
        return self.delay + self.attack + self.sustain + self.release

    @property
    def initial_value(self) -> float:
        return self.start_value

    # @cached_property
    # def final_value(self) -> float:
    #     if self.release > 0:
    #         return self.start_value
    #     else:
    #         return self.end_value

    @property
    def duration(self) -> int:
        return self.cycle_duration * self.cycles

    def calc(self, n: int) -> float:

        while n - 1 > self.cycle_duration:
            n -= self.cycle_duration

        if n <= self.delay:
            return self.start_value

        if n <= self.delay + self.attack + 1:
            return linear(
                n,
                self.delay + 1,
                self.delay + self.attack + 1,
                self.start_value,
                self.end_value,
            )

        if n <= self.delay + self.attack + self.sustain + 1:
            return self.end_value

        if n <= self.delay + self.attack + self.sustain + self.release + 1:
            return linear(
                n,
                self.delay + self.attack + self.sustain + 1,
                self.delay + self.attack + self.sustain + self.release + 1,
                self.end_value,
                self.start_value,
            )

        raise RuntimeError("unhandled n value")
