import enum


class SleepType(enum.Enum):
    WAKE = 0
    NREM = 1
    REM = 2


if __name__ == "__main__":
    print(len(SleepType))
