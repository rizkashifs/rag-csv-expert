import logging


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


logger = logging.getLogger("rag_csv_expert")
