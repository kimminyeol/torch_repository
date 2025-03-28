# extractor/extractor.py

import csv, json
from utils import load_prompt, call_openai
from parser import parse_entities_relations, parse_claims

TUPLE_DELIM = "|"
RECORD_DELIM = "\n"
COMPLETE_TOKEN = "### END ###"

def main():
    # Load prompts as plain strings (not Template)
    entity_prompt_str = load_prompt("prompts/entity_relation_prompt.txt")
    claim_prompt_str = load_prompt("prompts/claim_prompt.txt")

    # Load chunked text
    with open("../chunker/co.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        chunks = list(reader)

    all_entities, all_relations, all_claims = [], [], []

    for chunk in chunks:
        text = chunk["text"]

        # Entity & Relation Extraction
        filled_entity_prompt = entity_prompt_str.format(
            entity_types="ORGANIZATION,PERSON,GEO",
            tuple_delimiter=TUPLE_DELIM,
            record_delimiter=RECORD_DELIM,
            completion_delimiter=COMPLETE_TOKEN,
            input_text=text
        )
        entity_response = call_openai(filled_entity_prompt)
        ents, rels = parse_entities_relations(entity_response, TUPLE_DELIM, RECORD_DELIM)
        all_entities.extend(ents)
        all_relations.extend(rels)
        print(f"   ✅ 엔터티 {len(ents)}개, 관계 {len(rels)}개 추출")


        # Claim Extraction
        filled_claim_prompt = claim_prompt_str.format(
            entity_specs="ORGANIZATION,PERSON,GEO",
            claim_description="legal, suspicious, or controversial actions",
            tuple_delimiter=TUPLE_DELIM,
            record_delimiter=RECORD_DELIM,
            completion_delimiter=COMPLETE_TOKEN,
            input_text=text
        )
        claim_response = call_openai(filled_claim_prompt)
        claims = parse_claims(claim_response, TUPLE_DELIM, RECORD_DELIM)
        all_claims.extend(claims)
        
        print(f"   ✅ 클레임 {len(claims)}개 추출 완료")

    # Save outputs
    json.dump(all_entities, open("entities.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(all_relations, open("relations.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(all_claims, open("claims.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("✅ 저장 완료: entities.json / relations.json / claims.json")

if __name__ == "__main__":
    main()