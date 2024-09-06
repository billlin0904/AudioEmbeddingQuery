# Audio Embedding Database API

This repository contains a FastAPI-based API for extracting audio embeddings using a pre-trained PANNs model. It stores embeddings in ChromaDB for efficient similarity search and also performs audio track separation (vocals and instrumentals) before embedding extraction.

## Features

- **Audio Embedding**: Extracts embeddings using PANNs inference model.
- **Audio Separation**: Separates vocals and instrumentals using the UVR5 model before embedding.
- **Database Storage**: Stores embeddings in ChromaDB for fast retrieval and querying.
- **Similarity Search**: Queries ChromaDB to find similar audio based on embeddings.

## Tech Stack

- **FastAPI**: To build and serve the API.
- **PyTorch**: For PANNs model inference and GPU support.
- **Librosa**: For audio file loading and processing.
- **ChromaDB**: For storing and querying embeddings.
- **Torch**: Used to ensure compatibility with GPU/CPU.
- **UVR5**: For separating audio into vocals and instrumentals.

## Installation

### Prerequisites

- Python 3.8+
- CUDA (for GPU support, optional)

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/audio-embedding-database.git
    cd audio-embedding-database
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the UVR5 model weights and place the `2_HP-UVR.pth` file in the `uvr5_weights/` directory.

5. (Optional) Configure ChromaDB settings in the `AudioEmbeddingDatabase` class for persistent storage.

## Running the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
