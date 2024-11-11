from intervaltree import Interval, IntervalTree
from datetime import datetime, timedelta
from enum import Enum
import random


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Job:
    name: str
    priority: Priority

    def __init__(self, name: str, priority: Priority):
        self.name = name
        self.priority = priority

    def __str__(self):
        return f'{self.name} P{self.priority.value}'


def generate_interval(name, year, month, day, hour, minute, durationInMinutes, priority):
    start = datetime(year, month, day, hour, minute)
    end = start + timedelta(minutes=durationInMinutes)
    job = Job(name, priority)
    return Interval(start, end, job)


def generate_interval_today(name, hour, minute, durationInMinutes, priority):
    now = datetime.now()
    return generate_interval(name, now.year, now.month, now.day, hour, minute, durationInMinutes, priority)


def get_intervals_small():
    return [
        generate_interval_today('Job 01', 12, 00, 30, Priority.HIGH),
        generate_interval_today('Job 02', 12, 15, 60, Priority.MEDIUM),
        generate_interval_today('Job 03', 12, 45, 30, Priority.LOW),
        generate_interval_today('Job 04', 13,  0, 30, Priority.HIGH),
        generate_interval_today('Job 05', 13,  0, 30, Priority.LOW),
        generate_interval_today('Job 06', 13, 15, 30, Priority.MEDIUM),
        generate_interval_today('Job 07', 13, 15, 30, Priority.MEDIUM),
        generate_interval_today('Job 08', 13, 15, 30, Priority.HIGH),
        generate_interval_today('Job 09', 13, 15, 45, Priority.LOW),
        generate_interval_today('Job 10', 13, 15, 60, Priority.HIGH),
        generate_interval_today('Job 11', 13, 30, 75, Priority.MEDIUM),
        generate_interval_today('Job 12', 14,  0, 75, Priority.LOW),
        generate_interval_today('Job 13', 15, 15, 60, Priority.LOW),
        generate_interval_today('Job 14', 15, 45, 90, Priority.MEDIUM),
        generate_interval_today('Job 15', 17,  0, 15, Priority.MEDIUM)
    ]


def get_intervals_large():
    t = datetime.now()
    end = t + timedelta(days=1)

    intervals = []
    counter = 0

    while t < end:
        priority: Priority
        name: str

        choice = random.randint(0, 2)
        if choice == 1:
            priority = Priority.LOW
        elif choice == 2:
            priority = Priority.MEDIUM
        else:
            priority = Priority.HIGH

        choice = random.randint(0, 2)
        if choice == 1:
            durationInMinutes = 10
        elif choice == 2:
            durationInMinutes = 30
        else:
            durationInMinutes = 60

        name = f'Job {counter} P{priority.value}'

        interval = generate_interval(
            name,
            t.year,
            t.month,
            t.day,
            t.hour,
            t.minute,
            durationInMinutes,
            priority
        )
        intervals.append(interval)

        counter += 1
        t += timedelta(minutes=random.randint(0, 30))

    return intervals
