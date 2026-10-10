"""Minimal scheduler decision reproducer for the long-first/short-second case."""

from dataclasses import dataclass


@dataclass
class Request:
    name: str
    remaining: int


def break_policy(queue: list[Request], token_budget: int) -> list[str]:
    scheduled = []
    for request in queue:
        if request.remaining > token_budget:
            break
        scheduled.append(request.name)
        token_budget -= request.remaining
    return scheduled


def skip_policy(queue: list[Request], token_budget: int) -> list[str]:
    scheduled = []
    skipped = []
    for request in queue:
        if request.remaining > token_budget:
            skipped.append(request)
            continue
        scheduled.append(request.name)
        token_budget -= request.remaining
    return scheduled


if __name__ == "__main__":
    workload = [Request("long-4096", 4096), Request("short-128", 128)]
    print({"token_budget": 512, "break": break_policy(workload, 512), "skip": skip_policy(workload, 512)})
