import pandas as pd 
import numpy as np 
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
import os

# Load environment variables
load_dotenv()
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

# Load book metadata
books = pd.read_csv("./notebooks/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"] == "&fife=w800",
    "no_cover.jpg",
    books["large_thumbnail"]
)

# Load embedding model
embedding = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')

# Load and split documents
raw_documents = TextLoader("./notebooks/tagged_description.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
documents = text_splitter.split_documents(raw_documents)

# Create vector DB
db_books = Chroma.from_documents(documents, embedding)

# Recommendation engine
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = []
    for rec in recs:
        try:
            isbn = int(rec.page_content.strip('"').split()[0])
            books_list.append(isbn)
        except:
            continue

    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"] if isinstance(row["description"], str) else ""
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..." if truncated_desc_split else ""

        authors_split = row["authors"].split(";") if isinstance(row["authors"], str) else []
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"] if isinstance(row["authors"], str) else "Unknown"

        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        image_url = row["large_thumbnail"] if isinstance(row["large_thumbnail"], str) else "no_cover.jpg"
        # RETURN tuple (image_url, caption) NOT dict
        results.append((image_url, caption))

    return results


# UI setup
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown(
        """
        # 📚 Semantic Book Recommender
        _Find your next favorite book using the power of embeddings and vibes._
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="📖 Describe what you're looking for",
                placeholder="e.g., A coming-of-age story with magical realism...",
                lines=3,
                max_lines=5,
            )
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📂 Genre",
                value="All",
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Emotional Tone",
                value="All",
            )
            submit_button = gr.Button("✨ Get Recommendations", size="lg")

    gr.Markdown("---")
    gr.Markdown("## 🔍 Top Picks for You")

    output = gr.Gallery(label="Recommended Books", columns=4, rows=4, show_label=True)


    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()
