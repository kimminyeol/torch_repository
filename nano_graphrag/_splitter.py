from typing import List, Optional, Union, Literal

'''
1단계 
Document -> chunk
'''

'''
토큰 단위로 입력을 받아-> seperator 기준으로 나눔 -> 
'''
class SeparatorSplitter:
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
    ):
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

#   전체 chunking 함수 
    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

# sperator 리스트를 기준으로 토큰을 나눈다. 
# 토큰을 받고 sperators에 있는 토큰을 기준으로 나눈다.
    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        splits = []
        current_split = []
        i = 0
        while i < len(tokens):
            separator_found = False
            # 여러 개의 separator를 확인한다.
            for separator in self._separators:
                # separator가 등장할 경우
                if tokens[i:i+len(separator)] == separator:
                    # 현 조각의 끝에 seperator 추가
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    # split에 현재 split을 추가
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    # 다음 조각에 start를 붙임 
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    i += len(separator)
                    separator_found = True
                    break
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        if current_split:
            splits.append(current_split)
        return [s for s in splits if s]

# seperator를 기준으로 나눈 작은 조각들을 chunk 단위로 합침, 
# oeverlap이 있을 경우 중복 허용하여 청크 합침 
    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        if not splits:
            return []

        merged_splits = []
        current_chunk = []

        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                current_chunk.extend(split)
            else:
                merged_splits.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_splits.append(current_chunk)

        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])

        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        
        return merged_splits

# 청킹하는 함수 
# 청킹하는 함수는 청크의 길이와 오버랩을 고려하여 청크를 나눈다.
    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        result = []
        for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:  # 只有当 chunk 长度大于 overlap 时才添加
                result.append(new_chunk)
        return result


# 청크 오버랩을 강제하는 함수
    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                overlap = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap + chunk
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[:self._chunk_size]
                result.append(new_chunk)
        return result

