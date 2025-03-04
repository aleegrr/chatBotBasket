import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import io
import re
import requests
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter


def pdf_to_markdown(pdf_path):

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}

    response = requests.get(url=pdf_path, headers=headers, timeout=120)
    on_fly_mem_obj = io.BytesIO(response.content)
    pdf_document = PdfReader(on_fly_mem_obj)
    # Initialize the Markdown content
    markdown_content = ""

    # Iterate through each page of the PDF
    for page_number in range(len(pdf_document.pages)):
        page = pdf_document.pages[page_number]

        # Extract text from the page
        text = page.extract_text("text")

        # Process the text (you can customize this part based on your needs)
        soup = BeautifulSoup(text, "html.parser")
        formatted_text = soup.get_text(separator="\n")

        # Append the processed text to the Markdown content
        markdown_content += f"\n# Page {page_number + 1}\n\n{formatted_text}\n"

    return markdown_content


# Example usage
pdf_path = ""
markdown_content = pdf_to_markdown(pdf_path)

# Save the Markdown content to a file
with open("output.md", "w", encoding="utf-8") as markdown_file:
    markdown_file.write(markdown_content)