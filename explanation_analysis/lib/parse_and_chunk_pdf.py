import fitz  # PyMuPDF
import tiktoken
from more_itertools import unique_everseen

def pdf_to_text(input_file):
    # Open the provided PDF file
    doc = fitz.open(input_file)
    # Initialize containers for categorized text
    page_number_list, body_list = [], []
    # Categorize text blocks based on the most common font size
    for page in doc:
        page_number_list.append(page.get_label() if page.get_label() else page.number)
        body_list.append(page.get_text("text"))
    doc.close()
    return page_number_list, body_list

def chunk_text(tokens, chunk_size, chunk_overlap):
    chunks = []
    index = 0
    while index < len(tokens):
        end_index = index + chunk_size
        chunks.append(tokens[index:end_index])
        index += chunk_size - chunk_overlap
    return chunks

def count_tokens(text, model="gpt-4"):
    tiktoken_encoding = tiktoken.encoding_for_model(model)
    return len(tiktoken_encoding.encode(text))

def parse_and_chunk_pdf(file_path, chunk_size=800, chunk_overlap=400, model="gpt-4"):
    page_number_list, body_list = pdf_to_text(file_path)

    tiktoken_encoding = tiktoken.encoding_for_model(model)
    tokens = [
        (page_id,token)
        for page_id,text in zip(page_number_list, body_list)
        for token in tiktoken_encoding.encode(text)
    ]
    
    chunks = chunk_text(tokens, chunk_size, chunk_overlap)

    page_chunks = []
    for chunk in chunks:
        page_id_list, token_list = zip(*chunk)
        page_id_list = list(unique_everseen(page_id_list))
        page_id = f'{page_id_list[0]} - {page_id_list[-1]}' if len(page_id_list) > 1 else page_id_list[0]
        text = tiktoken_encoding.decode(token_list)
        page_chunks.append((page_id,text))
    return page_chunks

if __name__ == "__main__":
    # Example usage
    file_path = "../../textbooks/design_patterns/[2019]Dive Into DESIGN PATTERNS.pdf"
    chunks_with_pages = parse_and_chunk_pdf(file_path)
    for page, text in chunks_with_pages:
        print(f"Page {page}: {text}")
