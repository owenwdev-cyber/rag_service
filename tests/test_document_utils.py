"""
documents_utils.py
Test core functions: document loading, text reading, MD5 fingerprint, cache I/O
"""
import common

from src.rag_service.utils.document_utils import DocumentLoader


def test_load_documents():
    print("\n====== Test: Load Documents ======")
    try:
        docs = DocumentLoader.load_documents_from_folder()
        print(f"Successfully loaded {len(docs)} documents")

        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Filename: {doc['filename']}")
            print(f"Language: {doc['language']}")
            print(f"Doc2Query: {doc['enable_doc2query']}")
            print(f"Content length: {len(doc['content'])} chars")

    except Exception as e:
        print(f"Load failed: {e}")


def test_md5_and_cache():
    print("\n====== Test: MD5 & Cache ======")
    try:
        docs = DocumentLoader.load_documents_from_folder()
        if not docs:
            print("No documents available")
            return

        doc = docs[0]
        filename = doc["filename"]
        content = doc["content"]

        md5 = DocumentLoader.get_file_md5(content)
        print(f"Filename: {filename}")
        print(f"MD5: {md5}")

        fake_chunks = [content[:200], "test chunk 2"]
        DocumentLoader.save_chunks_to_cache(filename, md5, fake_chunks)

        cached = DocumentLoader.load_chunks_from_cache(filename, md5)
        if cached:
            print(f"Cache loaded successfully, chunk count: {len(cached)}")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # test_load_documents()
    # test_md5_and_cache()
    DocumentLoader.clear_chunk_cache()
    print("\nTest completed!")