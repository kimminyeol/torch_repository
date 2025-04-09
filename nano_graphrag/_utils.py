import asyncio
import html
import json
import logging
import os
import re
import numbers
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union

import numpy as np
import tiktoken

logger = logging.getLogger("nano-graphrag")
ENCODER = None

'''
LLM 결과 파싱
텍스트 토큰화 
캐시 키
json parsing
json encoding, decoding
json file read, write
json object
json object to string
json string to object
json object to file

'''

# asyncio 이벤트 루프를 항상 가져오는 함수
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# 문자열에서 첫 번째 완전한 JSON 객체를 추출하는 함수
def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start:i+1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None

#  문자열 값을 적절한 유형(int, float, bool, None 또는 문자열로 유지)으로 변환하는 함수
def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist

# 비정형 JSON 문자열에서 키 값을 추출하는 함수
def extract_values_from_json(json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}
    
    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'
    
    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
    
    return extracted_values

# 사용자의 응답을 JSON으로 변환하는 함수
def convert_response_to_json(response: str) -> dict:
    """Convert response string to JSON, with error handling and fallback to non-standard JSON extraction."""
    prediction_json = extract_first_complete_json(response)
    
    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)
    
    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")
    
    return prediction_json



# 문자열을 token 단위로 인코딩하는 함수
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

# token 단위로 인코딩된 문자열을 디코딩하는 함수
def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content

# 리스트의 각 요소를 token 단위로 인코딩하는 함수
def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

# 리스트의 각 요소를 token 단위로 인코딩하여 최대 token 크기를 초과하는지 확인하는 함수
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

# JSON 객체를 파일에 쓰는 함수
def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

# JSON 파일을 읽는 함수
def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


# 사용자의 프롬프트와 생성된 콘텐츠를 OpenAI 메시지 형식으로 패킹하는 함수
# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(prompt: str, generated_content: str, using_amazon_bedrock: bool):
    if using_amazon_bedrock:
        return [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": generated_content}]},
        ]
    else:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": generated_content},
        ]

def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()

# 다양한 문자로 문자열 분할 
def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

# 문자열을 따옴표로 감싸기
def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content) 
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'

# 리스트를 csv로
def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
# html escape 제거, 제어 문자 제거 
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------

@dataclass
# 임베딩 함수의 속성을 저장하는 클래스
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
# 비동기 함수 호출 횟수 제한
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro

# 임베딩 함수에 속성을 추가
def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
