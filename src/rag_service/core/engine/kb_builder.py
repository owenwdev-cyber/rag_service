"""
Knowledge Base Building Engine: Full Build / Incremental Update
"""
import logging
from src.rag_service.core.settings import settings
from src.rag_service.utils.document_utils import DocumentLoader
from src.rag_service.core.engine.kb_manager import KBManager

logger = logging.getLogger(__name__)


class KBBuilder:
    def __init__(self):
        self.manager = KBManager()

    def auto_build(self, folder=None, force_rebuild=False):
        """Automatically build or update knowledge base based on current state"""
        if folder is None:
            folder = settings.document.documents_folder

        if force_rebuild or not self.manager.state_exist():
            logger.info("Starting full knowledge base build")
            self._full_build(folder)
        else:
            logger.info("Starting incremental knowledge base update")
            self._incremental_update(folder)

    def _full_build(self, folder):
        """Build the entire knowledge base from scratch"""
        docs = DocumentLoader.load_documents_from_folder(folder)
        self.manager.reset()

        total_docs = len(docs)
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting full knowledge base build")
        logger.info(f"{'='*60}")

        for idx, doc in enumerate(docs, 1):
            fname = doc["filename"]
            content = doc["content"]
            enable_doc2query = doc["enable_doc2query"]
            language = doc.get("language", "zh")
            md5 = DocumentLoader.get_file_md5(content)

            logger.info(f"[{idx}/{total_docs}] Processing file: {fname}")
            self.manager.add_file(fname, content, md5, enable_doc2query, language)

        self.manager.build_retriever()
        self.manager.save()
        logger.info("Full knowledge base build completed")

    def _incremental_update(self, folder):
        """Update knowledge base incrementally by detecting file changes"""
        if not self.manager.load():
            logger.warning("No existing index found, automatically switching to full build")
            self._full_build(folder)
            return

        docs = DocumentLoader.load_documents_from_folder(folder)
        current_files = {d["filename"]: d for d in docs}

        for fname in list(self.manager.file_md5.keys()):
            if fname not in current_files:
                logger.info(f"Removing file: {fname}")
                self.manager.remove_file(fname)

        total_updates = sum(
            1 for fname, doc in current_files.items()
            if fname not in self.manager.file_md5 or
            self.manager.file_md5[fname] != DocumentLoader.get_file_md5(doc["content"])
        )
        update_count = 0

        for fname, doc in current_files.items():
            new_md5 = DocumentLoader.get_file_md5(doc["content"])
            if fname in self.manager.file_md5 and self.manager.file_md5[fname] == new_md5:
                continue

            update_count += 1
            logger.info(f"Updating file [{update_count}/{total_updates}]: {fname}")
            self.manager.remove_file(fname)
            self.manager.add_file(
                fname, doc["content"], new_md5,
                doc["enable_doc2query"], doc.get("language", "zh")
            )

        self.manager.save()
        logger.info("Incremental update completed")