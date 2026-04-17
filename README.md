# Project-2030-MyAI-Future-Final-Assessmenta# CourseMate AI – Course Materials Finder Assistant (RAG)

## Overview
CourseMate AI helps students find the right course materials (lecture slides, notes, assignment briefs) in seconds.
Users upload PDFs and ask questions in natural language. The system uses Retrieval-Augmented Generation (RAG) to produce grounded answers with citations.

## Problem Statement
Course materials are scattered across LMS, drives, chats, and multiple PDFs. Students waste time searching and may miss key requirements (deadlines, rubrics). CourseMate AI provides semantic retrieval + AI answers backed by document citations.

## Target Users
- Students (primary)
- Tutors/Lecturers (secondary)

## Solution Overview
1) Upload PDFs
2) Parse + chunk
3) Create embeddings
4) Retrieve Top-K relevant chunks for a question
5) Gemini generates the final answer using only retrieved context + shows citations

## Tech Stack
- Streamlit (UI)
- pypdf (PDF parsing)
- Google GenAI SDK (Gemini API)
- Embeddings: gemini-embedding-001
- LLM: gemini-3-flash-preview
- Similarity: cosine similarity (NumPy)

## Architecture Diagram
(You can paste the Mermaid diagram from the slides or below.)

```mermaid
flowchart LR
  U[Student] --> UI[Streamlit UI]
  UI --> P[PDF Parser] --> C[Chunker]
  C --> E[Embeddings: gemini-embedding-001]
  U -->|Question| Q[Query Embedding]
  E --> S[Similarity Search Top-K]
  S --> L[LLM: gemini-3-flash-preview]
  L --> A[Answer + Citations]
  A --> UI
