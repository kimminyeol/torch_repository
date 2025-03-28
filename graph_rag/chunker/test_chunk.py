import csv
from chunk_generator import chunk_document

def read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def save_chunks_to_csv(chunks, output_path="chunked_output.csv"):
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["chunk_id", "start_token", "end_token", "text"])  # 헤더
        for chunk in chunks:
            writer.writerow([chunk.id, chunk.start_token, chunk.end_token, chunk.text])

def main():
    # 1. 문서 읽기
    text = read_file("test.txt")

    # 2. 청크 분할
    chunks = chunk_document(text)

    # 3. CSV로 저장
    save_chunks_to_csv(chunks, output_path="chunked_output.csv")
    print("✅ 청크가 chunked_output.csv에 저장되었습니다.")

if __name__ == "__main__":
    main()