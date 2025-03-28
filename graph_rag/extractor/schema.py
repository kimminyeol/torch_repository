# extractor/schema.py

from dataclasses import dataclass

@dataclass
class Entity:
    name: str
    type: str
    description: str

@dataclass
class Relation:
    source: str
    target: str
    description: str
    strength: int

@dataclass
class Claim:
    subject: str
    object: str
    claim_type: str
    status: str
    start_date: str
    end_date: str
    description: str
    source_text: str