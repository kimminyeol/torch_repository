from typing import List
from chunk_schema import Chunk
from tokenizer_utils import get_tokenizer, slice_text_by_tokens
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOKENIZER_MODEL

def chunk_document(text: str) -> List[Chunk]:
    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    
    chunks = []
    start = 0
    chunk_id = 0

    while start < total_tokens:
        end = min(start + CHUNK_SIZE, total_tokens)
        chunk_text = slice_text_by_tokens(text, tokenizer, start, end)
        chunks.append(Chunk(
            id=chunk_id,
            text=chunk_text,
            start_token=start,
            end_token=end
        ))
        chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP  # 슬라이딩 윈도우

    return chunks