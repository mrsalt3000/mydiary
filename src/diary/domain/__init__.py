"""Domain use-case layer of Local Diary.

Business logic lives in per-area subpackages (``chat``, later ``audio``,
``exports``, ``llm``, ``pipelines``); HTTP routers stay thin and reach the
database only through domain services and repositories.
"""
