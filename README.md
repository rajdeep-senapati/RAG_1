# Retrieval-Augmented Generation (RAG) with Open-Source LLMs 

## Overview

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using open-source large language models (LLMs) from Hugging Face, integrated with LangChain and FAISS for efficient document search and grounded question answering. The solution is designed for ease of use in Google Colab, leveraging GPU acceleration for scalable, interactive AI applications.

---

## Features

- **RAG Pipeline:** Combines semantic document retrieval with generative AI for context-aware answers.
- **Open-Source Models:** Utilizes models like Mistral-7B or Gemma-2B for high-quality, cost-effective inference.
- **Efficient Vector Search:** Employs FAISS for fast, scalable similarity search over large document sets.
- **Memory Optimization:** Supports 4-bit quantization with bitsandbytes for running large models on limited GPU resources.
- **Supports Multiple Data Types:** Easily ingest PDFs and other text sources.
- **Clean Output:** Returns only the final answer and its source references, suppressing intermediate reasoning and system messages.

---

## Setup Instructions

1. **Clone or open the notebook in Google Colab.**
2. **Enable GPU:**  
   Go to `Runtime > Change runtime type > Hardware accelerator: GPU`.
3. **Install dependencies:**  
   All required packages are installed automatically in the first code cell.
4. **Upload or specify your documents:**  
   By default, the notebook demonstrates PDF ingestion from a URL; you can upload your own files.
5. **Ask questions:**  
   Enter your queries in the provided cell and receive concise, source-grounded answers.

---

## Usage Example

**query = "What is retrieval-augmented generation?"**
**response = qa_chain.invoke({"query": query})**


**Extract only the final answer (removes chain of thought)**
import re
raw_output = response["result"]
match = re.search(r"Helpful Answer:\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)
answer = match.group(1).strip() if match else raw_output.strip().split('\n')[-1]
print(answer)
for doc in response["source_documents"]:
print(f"{doc.metadata['source']} (page {doc.metadata.get('page', '?')})")


**Output:**
Retrieval-augmented generation refers to a method used by large language models where they first retrieve relevant information from external sources using a retrieval system, and then generate responses based on this combined data. This approach can help improve the accuracy and relevance of the generated responses compared to solely relying on the model's internal knowledge.

https://arxiv.org/pdf/2303.08774.pdf (page 55)
https://arxiv.org/pdf/2303.08774.pdf (page 75)
https://arxiv.org/pdf/2303.08774.pdf (page 71)

---

## Requirements

- **Google Colab** (with GPU runtime enabled)
- **Python 3.10+**
- **Libraries:**  
  - `transformers`  
  - `langchain`  
  - `sentence-transformers`  
  - `faiss-cpu`  
  - `bitsandbytes`  
  - `accelerate`  
  - `pypdf`

All dependencies are installed automatically in the notebook.

---

## Customization

- **Model Selection:** Swap in your preferred Hugging Face LLM and embedding model.
- **Data Sources:** Easily extend to ingest websites, text files, or other formats.
- **Prompting:** The notebook includes logic to filter out chain-of-thought and system messages for clean, user-focused answers.

---

## Troubleshooting

- **No GPU Detected:** Ensure Colab runtime is set to GPU and restart the runtime.
- **bitsandbytes or CUDA Errors:** The notebook installs the latest compatible versions; if issues persist, restart the runtime and rerun all cells.
- **Output Includes Unwanted Text:** The provided post-processing code ensures only the answer and sources are shown.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

**For questions or contributions, please open an issue or pull request.**
