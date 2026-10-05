from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/test.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

for page in documents[:2]:
    print(page.page_content[:500])