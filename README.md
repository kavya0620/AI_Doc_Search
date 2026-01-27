# AI Document Search - RAG Chatbot

A powerful RAG (Retrieval-Augmented Generation) chatbot that allows you to upload documents and ask questions about their content using AI.

## Features

- **Multi-document support** - Upload and process multiple PDFs, Word docs, PowerPoint, text files, and images
- **Intelligent search** - Advanced search with exact phrase matching, BM25 keyword search, and fallback strategies
- **Multi-modal processing** - Supports text extraction from images using OCR
- **Session-based** - Documents are processed for the current session only (no persistent storage)
- **Export capabilities** - Export chat history and analytics
- **User authentication** - Role-based access control
- **Real-time chat** - Interactive Q&A interface

## Supported File Types

- PDF (.pdf)
- Text files (.txt)
- Word documents (.docx)
- PowerPoint presentations (.pptx)
- Images (.png, .jpg, .jpeg)

## Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Google Gemini API
- **Document Processing**: LangChain
- **Search**: BM25 (rank-bm25)
- **OCR**: Tesseract
- **Multi-modal**: Transformers (BLIP, CLIP)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AI_Doc_Search.git
cd AI_Doc_Search
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
```
Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

4. **Get Google API Key**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Copy and paste it into your `.env` file

## Usage

1. **Run the application**
```bash
streamlit run ai_document_app_enhanced.py
```

2. **Login**
   - Use demo credentials: `admin/admin123`, `faculty/faculty123`, or `student/student123`

3. **Upload documents**
   - Use the sidebar to upload one or multiple documents
   - Click "Process Documents" to prepare them for querying

4. **Ask questions**
   - Type your questions in the chat interface
   - Get AI-powered answers based on your document content

5. **Export data**
   - Export chat history and analytics using the export buttons

## How It Works

1. **Document Processing**: Documents are chunked into smaller pieces for better retrieval
2. **Indexing**: BM25 indexing creates searchable representations
3. **Query Processing**: User questions are matched against document chunks
4. **AI Generation**: Google Gemini generates answers based on retrieved content
5. **Response**: Users get contextual answers with source references

## Configuration

### Chunk Settings
- `CHUNK_SIZE = 1000` - Size of document chunks
- `CHUNK_OVERLAP = 100` - Overlap between chunks

### Models
- Primary: `gemini-2.5-flash`
- Fallback: `gemini-2.0-flash`, `gemini-flash-latest`

## Project Structure

```
AI_Doc_Search/
├── ai_document_app_enhanced.py    # Main application
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── data/                         # Sample documents
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Troubleshooting

### API Issues
- Ensure your Google API key is valid
- Check if you have quota remaining
- Verify the Generative Language API is enabled

### Document Processing
- Supported file types: PDF, TXT, DOCX, PPTX, PNG, JPG, JPEG
- Large files may take longer to process
- Clear documents and reload if having issues

### Search Issues
- Try rephrasing your question
- Use keywords from the document
- Check if the information exists in uploaded documents
