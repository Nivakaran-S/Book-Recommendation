# Semantic Book Recommender System
Welcome to the Semantic Book Recommender System, an innovative application that leverages natural language processing (NLP) and semantic search to deliver personalized book recommendations based on user queries, book categories, and emotional tones. Built with a modern tech stack and a user-friendly interface, this project showcases advanced machine learning techniques and a passion for creating intuitive, impactful solutions.

## 🌟 Project Overview
This project is a semantic book recommendation engine that allows users to input a description of a book they’re looking for, select a category (e.g., Fiction, Nonfiction), and choose an emotional tone (e.g., Happy, Suspenseful). Using state-of-the-art NLP embeddings and a vector database, the system retrieves and ranks book recommendations that align with the user’s preferences. The results are displayed in a visually appealing gallery format, complete with book covers and concise descriptions.
The project demonstrates proficiency in Python, machine learning, NLP, and web development, making it a robust showcase of full-stack data science skills.

## 🚀 Features
- Semantic Search: Utilizes Hugging Face’s all-MiniLM-L6-v2 model for generating embeddings and Chroma vector database for efficient similarity search.
- Customizable Recommendations: Filter books by category and sort by emotional tone (e.g., Happy, Sad, Suspenseful) based on precomputed sentiment scores.
- Interactive UI: Built with Gradio and themed with a sleek Glass design for an engaging user experience.
- Data Processing: Leverages Pandas for efficient data manipulation and preprocessing of a book dataset with rich metadata (e.g., ISBN, title, authors, description, emotions).
- Scalable Architecture: Modular codebase with clean separation of data processing, model integration, and frontend logic, ensuring maintainability and extensibility.
- Environment Management: Securely handles API keys using python-dotenv for seamless integration with external services like Hugging Face.

## 🛠️ Tech Stack
- Programming Language: Python
- Data Processing: Pandas, NumPy
- NLP & Embeddings: LangChain, HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- Vector Database: Chroma
- Frontend: Gradio (with Glass theme)
- Environment Management: python-dotenv
- Dataset: Custom books_with_emotions.csv dataset containing book metadata and precomputed emotional scores

## 📂 Project Structure
```
├── notebooks/
│   ├── books_with_emotions.csv   # Dataset with book metadata and emotions
│   ├── tagged_description.txt    # Preprocessed book descriptions for embedding
├── .env                          # Environment file for API keys
├── main.py                       # Main application script
├── README.md                     # Project documentation
└── requirements.txt              # Project dependencies
```

## 🔧 Installation
Follow these steps to set up the project locally:

### 1. Clone the Repository:
```bash
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
```

### 2. Set Up a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables:
Create a .env file in the project root.
```bash
Add your Hugging Face API key:HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### 5. Run the Application:
```bash
python main.py
```

This will launch the Gradio interface in your default web browser, where you can input queries and explore recommendations.


## 📖 Usage
- Input a Query: Enter a description of the book you’re looking for (e.g., "A story about forgiveness").
- Select a Category: Choose a category from the dropdown (e.g., Fiction, Nonfiction, or All).
- Choose an Emotional Tone: Select a tone (e.g., Happy, Sad, Suspenseful, or All).
- Get Recommendations: Click the "Find recommendations" button to view a gallery of book covers with titles, authors, and truncated descriptions.

## 🎯 How It Works
- Data Loading: The system loads a preprocessed dataset (books_with_emotions.csv) containing book metadata and emotional scores (e.g., joy, sadness).
- Text Embedding: Book descriptions from tagged_description.txt are split into chunks and embedded using Hugging Face’s all-MiniLM-L6-v2 model.
- Vector Search: The Chroma vector database performs similarity searches to find books matching the user’s query.
- Filtering & Sorting: Results are filtered by category and sorted by the selected emotional tone, leveraging precomputed sentiment scores.
- Frontend Rendering: The Gradio interface displays the top recommendations in a gallery format with book covers and concise captions.

## 🌍 Future Enhancements
- Advanced Filtering: Add more filters like publication year, author, or rating.
- Real-Time Sentiment Analysis: Integrate a live sentiment analysis model to compute emotional tones dynamically.
- Improved UI/UX: Enhance the Gradio interface with custom styling and interactive features like book previews.
- API Integration: Expand the system to fetch real-time book data from external APIs (e.g., Goodreads, Google Books).
- Model Optimization: Experiment with larger or domain-specific embedding models for improved semantic search accuracy.

## 🙌 Acknowledgments
- Hugging Face: For providing the all-MiniLM-L6-v2 model for embeddings.
- LangChain: For enabling seamless integration of NLP tools and vector databases.
- Gradio: For powering the interactive web interface.
- Chroma: For efficient vector storage and similarity search.

