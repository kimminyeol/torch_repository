from dataclasses import dataclass

@dataclass
class Chunk:
    id: int
    text: str
    start_token: int
    end_token: int