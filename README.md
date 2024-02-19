# PDF Query

## Overview
This script provides a powerful tool for extracting text from PDF documents in a specified directory and answering questions based on the content of these documents. It utilizes [Retrieval Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401), with an SBERT vector index for efficient searching and a RAG-tuned language model for generating answers to queries.

## Requirements
- Python 3.x
- PyMuPDF: For PDF parsing.
- Tesseract OCR: For text extraction from PDFs.
- llmware: A package for interacting with language models.

## Installation
1. Ensure Python 3.x is installed on your system.
2. Install Tesseract OCR and note the installation path. Installation instructions can be found on the [Tesseract GitHub page](https://github.com/tesseract-ocr/tesseract).
3. Install required Python libraries: `pip install sentence-transformers llmware PyMuPDF`
4. Clone this repo to your local machine.

## Usage
The script is executed from the command line and takes several arguments:

- `--pdf-directory`: (Required) The path to the directory containing PDF files you want to query.
- `--tesseract-path`: (Required) The path to the Tesseract OCR executable. This is used in case a PDF does not yield enough text through PyMuPDF extraction.
- `--questions`: (Optional) A list of questions to be asked. If not provided, the script will prompt for questions interactively.

### Example Command

`python pdf_query.py --pdf-directory "/path/to/pdf_dir" --tesseract-path "/path/to/tesseract" --questions "What is the main topic of document 1?" "Who is the author of document 2?"`

If no questions are provided through the command line, the script enters an interactive mode, prompting the user to input questions manually.

## How It Works
1. **Index Building**: The script first reads all PDF documents in the provided directory, extracting their text using PyMuPDF or Tesseract OCR. It chunks those using a rolling overlapping window, and encodes those chunks to vectors for efficient querying.
2. **Query Processing**: It then either takes user-provided questions or enters an interactive mode for question input. It encodes the question using SBERT and then retrieves the most similar section of text from the indexed documents.
3. **Answer Generation**: Using a language model, the script generates an answer based on the context of the retrieved text section.