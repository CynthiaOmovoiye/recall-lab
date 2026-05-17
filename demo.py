from datetime import datetime
from pathlib import Path

from recall_lab.controls.sliding import SlidingWindowAgent
from recall_lab.memory.episodic import EpisodicLog, Exchange

turns = [
    "My favorite color is blue.",
    "I like reading history books.",
    "I am testing memory systems today.",
    "What is 2 + 2?",
    "What is my favorite color?",
]

agent = SlidingWindowAgent(window=2)
log = EpisodicLog(db_path=Path("data/demo/log.db"))

for i, user_turn in enumerate(turns, start=1):
    response = agent.respond(user_turn)
    log.append(
        Exchange(
            user=user_turn,
            agent=response,
            timestamp=datetime.utcnow(),
        )
    )
    print(f"\nTurn {i}")
    print("user:", user_turn)
    print("agent:", response)

today_rows = log.fetch_day(datetime.utcnow())

print("\nStored exchanges today:", len(today_rows))