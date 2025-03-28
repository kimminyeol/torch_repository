# extractor/parser.py

import re

def parse_entities_relations(text: str, tuple_delimiter="|", record_delimiter="\n"):
    entities, relations = [], []
    for line in text.strip().split(record_delimiter):
        if line.startswith("(\"entity\""):
            parts = re.findall(r'\("entity"\|(.+?)\|(.+?)\|(.+?)\)', line)
            if parts:
                name, type_, desc = parts[0]
                entities.append({
                    "name": name.strip(),
                    "type": type_.strip(),
                    "description": desc.strip()
                })
        elif line.startswith("(\"relationship\""):
            parts = re.findall(r'\("relationship"\|(.+?)\|(.+?)\|(.+?)\|(\d+)\)', line)
            if parts:
                src, tgt, desc, strength = parts[0]
                relations.append({
                    "source": src.strip(),
                    "target": tgt.strip(),
                    "description": desc.strip(),
                    "strength": int(strength)
                })
    return entities, relations


def parse_claims(text: str, tuple_delimiter="|", record_delimiter="\n"):
    claims = []
    for line in text.strip().split(record_delimiter):
        parts = re.findall(r'\((.+?)\)', line)
        if parts:
            fields = parts[0].split(tuple_delimiter)
            if len(fields) == 8:
                claims.append({
                    "subject": fields[0].strip(),
                    "object": fields[1].strip(),
                    "claim_type": fields[2].strip(),
                    "status": fields[3].strip(),
                    "start_date": fields[4].strip(),
                    "end_date": fields[5].strip(),
                    "description": fields[6].strip(),
                    "source_text": fields[7].strip(),
                })
    return claims