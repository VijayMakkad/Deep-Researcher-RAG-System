import os
import random
import traceback

# Lazy imports
try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    import ollama
except Exception:
    ollama = None

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Default configs
BACKEND = os.getenv("SUMMARIZER_BACKEND", "lexrank").lower()
DISTILBART_MODEL = "sshleifer/distilbart-cnn-12-6"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def make_generator(backend: str = None):
    """
    Load generator depending on backend.
    Ollama: no preload (called directly later).
    T5: DistilBART pipeline on CPU.
    LexRank: Extractive summarizer.
    """
    backend = (backend or BACKEND).lower()

    if backend == "lexrank":
        return LexRankSummarizer()

    if backend == "t5":
        if pipeline is None:
            print("⚠️ transformers.pipeline not available; fallback to LexRank.")
            return LexRankSummarizer()
        try:
            print(f"Loading DistilBART ({DISTILBART_MODEL}) on CPU...")
            return pipeline("summarization", model=DISTILBART_MODEL, device=-1)
        except Exception as e:
            print("⚠️ Failed to load DistilBART:", e)
            traceback.print_exc()
            return LexRankSummarizer()

    if backend == "ollama":
        if ollama is None:
            raise RuntimeError("❌ Ollama not installed. Install and run it first.")
        print(f"✅ Using Ollama model: {OLLAMA_MODEL}")
        return None

    # fallback default
    return LexRankSummarizer()


def _lexrank_summary(generator, text: str, sentence_count: int = 5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = generator(parser.document, sentences_count=sentence_count)
    return " ".join([str(s) for s in summ])


def generate_answer(query: str, contexts: list, generator,
                    max_length: int = 200, max_words: int | None = None,
                    backend: str = None, diverse: bool = False):
    """
    Generate answer using the requested backend.
    - lexrank: extractive
    - t5: abstractive (DistilBART)
    - ollama: strict, no fallback
    """
    backend = (backend or BACKEND).lower()
    combined = "\n\n".join(contexts)

    # ---------------- LEXRANK ----------------
    if backend == "lexrank":
        if diverse:
            outputs = []
            for i in range(3):
                sentence_count = random.choice([2, 3, 4, 5])
                outputs.append(_lexrank_summary(generator, combined, sentence_count))
            return "\n\n---\n\n".join(outputs)
        return _lexrank_summary(generator, combined, sentence_count=5)

    # ---------------- T5 / DistilBART ----------------
    if backend == "t5":
        try:
            prompt = f"Answer the query based on context:\nQuestion: {query}\n\nContext:\n{combined}\n\nProvide a concise summary."
            if len(prompt) > 3000:
                prompt = prompt[:3000]
            ml = max_length
            if max_words:
                ml = max_words * 2  # heuristic
            out = generator(prompt, max_length=ml, min_length=30, do_sample=False)
            return out[0]["summary_text"]
        except Exception as e:
            print("⚠️ DistilBART generation failed, falling back to LexRank:", e)
            return _lexrank_summary(LexRankSummarizer(), combined, sentence_count=5)

    # ---------------- OLLAMA (Strict) ----------------
    if backend == "ollama":
        if ollama is None:
            raise RuntimeError("❌ Ollama not available.")

        # word limit instruction
        word_instruction = ""
        if max_words:
            low = int(max_words * 0.9)
            high = int(max_words * 1.1)
            word_instruction = f"Write the answer in about {max_words} words. Ensure it's between {low} and {high} words."

        # structured academic prompt
        prompt = f"""
You are an expert research assistant.
Using only the provided context, produce a clear, well-structured academic answer.

❓ Question:
{query}

📚 Context:
{combined}

✍️ Guidelines for your answer:
1. Start with a crisp definition of the term or concept.
2. Explain the architecture, methodology, or approach in 2–3 sentences.
3. Present the main contributions, results, or findings as bullet points.
4. Highlight applications, strengths, or limitations if mentioned in context.
5. End with a 1–2 sentence conclusion summarizing its overall significance.
6. Maintain a formal, academic tone and avoid repetition.
7. Do not add any information outside of the provided context.

{word_instruction}
"""

        # Create Ollama client with custom host if specified
        ollama_host = os.getenv("OLLAMA_HOST")
        if ollama_host:
            client = ollama.Client(host=f"http://{ollama_host}")
            response = client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        else:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        
        answer = response["message"]["content"]

        # expansion if too short
        if max_words:
            wc = len(answer.split())
            if wc < int(max_words * 0.8):
                expand_prompt = f"The previous answer had {wc} words. Expand and refine it to ~{max_words} words while keeping it structured."
                if ollama_host:
                    response = client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": expand_prompt}])
                else:
                    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": expand_prompt}])
                answer = response["message"]["content"]

        return answer

    return "❌ Invalid backend."
