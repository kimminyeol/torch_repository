
import tiktoken

def get_tokenizer(model_name="gpt-3.5-turbo"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def slice_text_by_tokens(text, tokenizer, start_token, end_token):
    tokens = tokenizer.encode(text)
    sliced = tokens[start_token:end_token]
    return tokenizer.decode(sliced)