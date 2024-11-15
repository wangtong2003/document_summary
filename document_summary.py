import PyPDF2
import os
from docx import Document
import markdown
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import ollama
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text

def read_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def read_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return markdown.markdown(text)

def read_epub(file_path):
    book = epub.read_epub(file_path)
    text = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text += soup.get_text() + '\n'
    return text

def read_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    if file_extension == '.pdf':
        return read_pdf(file_path)
    elif file_extension == '.docx':
        return read_docx(file_path)
    elif file_extension == '.txt':
        return read_txt(file_path)
    elif file_extension == '.md':
        return read_md(file_path)
    elif file_extension == '.epub':
        return read_epub(file_path)
    else:
        raise ValueError("Unsupported file format")

def ollama_text(input, model='model="qwen2.5:3b'):#model='qwen2.5:32b-instruct'\model="qwen2.5:3b"
    prompt=f"""
    # System Instruction
You are a powerful AI assistant capable of reading text content from a JSON input and summarizing it. The purpose of the summary is to help users grasp the key points without reading the entire article.

[INSTRUCTIONS]
- The article will be provided in JSON with "page_num" and "text" as keys, page number and paging content as values in the <ARTICLE>XXX</ARTICLE>. Read through the article with all the content;
- Write a concise summary within 800 characters, capturing the essence of the article in one sentence;
- Outline the article in up to 10 points, with each point being less than 120 characters;
- Identify the page number(s) of where each point summarized from referring to "page_num", put the page number(s) at the end of each point. Avoid adding any page number to the summary;
- Utilize only factual information from the original text. Refrain from fabricating content;
- Please use Arabic numerals for point numbering, numbering does not need to be bold;
- Before finalizing, check the word count of the expanded text to ensure it is within the length limitation;
- If the expanded text exceeds the length limitation, trim redundant or overly detailed parts to meet the required length;
- Carefully verify that the page numbers in the output match the page numbers in the input. Make necessary corrections to ensure accuracy;
- The output should be in the language of the Chinese;
- Do not return using markdown format. For example, '##' formats the title, '**' bolds important content;
- Please follow the required format and do not include extraneous content;

<OUTPUT_FORMAT_1>
This is an example summary.
1.Point one.[P2]
2.Point two.[P2, P3]
...
</OUTPUT_FORMAT_1>

<OUTPUT_FORMAT_2>
这是一个示例总结。
1.要点一。[P1]
2.要点二。[P1, P2, P3]
...
</OUTPUT_FORMAT_2>

# prompt
ARTICLE_TO_SUMMARY:
<ARTICLE>
{input}
</ARTICLE>


**Note:**
1.Please repeatedly check and confirm that the page numbers in the output match the page numbers in the input. 
2.Note that the summary should not be marked with page numbers, but each core event needs to be marked with page numbers.
3.Please re-check, if you do not meet the above three points, please revise it immediately
"""
    client = ollama.Client(host='http://127.0.0.1:11434')
    response = client.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response['message']['content']


def process_document(file_path):
    document_content = read_document(file_path)
    result=ollama_text(document_content)
    return result


# 使用示例
file_path = r"E:\\第六周周报——superset可视化.pdf"
result=process_document(file_path)
print(result)

