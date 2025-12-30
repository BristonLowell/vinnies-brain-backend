import json
import os
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

from openai_client import embed_text

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/vinniesbrain"
)

ARTICLES_FILE = "vinnies_brain_articles.json"


def to_vector_literal(vec):
    """
    Convert a Python list of floats into a pgvector-compatible string.
    Example: [0.1, 0.2] -> "[0.1,0.2]"
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def normalize_article(item):
    """
    Ensures each article becomes a dict.
    Handles:
      - dict already
      - JSON string of dict: "{...}"
      - double-encoded JSON string: "\"{...}\""
    """
    tries = 0
    while isinstance(item, str) and tries < 3:
        s = item.strip()
        if not s:
            break
        if s[0] in ['{', '[', '"']:
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                break
        else:
            break
        tries += 1
    return item


def get_year_range(article: dict):
    """
    Your current article schema does not include per-article year min/max,
    but your scope is 2010–2025. We support multiple possible future shapes.
    Returns (years_min, years_max).
    """
    if "applies_to_years" in article and isinstance(article["applies_to_years"], dict):
        y = article["applies_to_years"]
        return int(y.get("min", 2010)), int(y.get("max", 2025))

    if "years" in article and isinstance(article["years"], dict):
        y = article["years"]
        return int(y.get("min", 2010)), int(y.get("max", 2025))

    if "years_min" in article and "years_max" in article:
        return int(article["years_min"]), int(article["years_max"])

    if "year_min" in article and "year_max" in article:
        return int(article["year_min"]), int(article["year_max"])

    # Default to your stated scope
    return 2010, 2025


def norm_list(x):
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    return []


def model_year_notes_to_list(myn):
    """
    Your schema uses model_year_notes as a dict:
      {"2010-2016": "...", "2017-2025": "..."}
    Convert to list of strings for storage + retrieval.
    """
    if isinstance(myn, dict):
        return [f"{k}: {v}" for k, v in myn.items() if str(v).strip()]
    if isinstance(myn, list):
        return norm_list(myn)
    return []


def make_retrieval_text(article: dict) -> str:
    ymin, ymax = get_year_range(article)

    title = article.get("title", "")
    category = article.get("category", "")
    severity = article.get("severity", "")

    # Your schema
    description = article.get("description", "")
    symptoms = norm_list(article.get("symptoms", []))
    initial_checks = norm_list(article.get("initial_checks", []))
    common_causes = norm_list(article.get("common_causes", []))
    confirmation = norm_list(article.get("confirmation", []))
    do_not = norm_list(article.get("do_not", []))
    common_sources = norm_list(article.get("common_sources", []))

    model_year_notes = model_year_notes_to_list(article.get("model_year_notes", {}))

    # escalation keys vary in your JSON
    stop_and_escalate = norm_list(article.get("stop_and_escalate", article.get("escalate_if", [])))

    recommended_action = article.get("recommended_action", "")

    # Build a few grounded clarifying questions (optional)
    clarifying_questions = []
    if symptoms:
        clarifying_questions.append("Which symptom(s) match what you’re seeing?")
    if confirmation:
        clarifying_questions.append("Do these confirmation signs apply?")
    if "Leak" in title or "leak" in description.lower():
        clarifying_questions.append("Does it happen after rain, washing, or both?")

    # Build steps for retrieval (store initial_checks in DB steps; this is for retrieval text only)
    steps_for_retrieval = []
    if initial_checks:
        steps_for_retrieval += [f"Initial check: {s}" for s in initial_checks]
    if common_causes:
        steps_for_retrieval += [f"Common cause to inspect: {c}" for c in common_causes]
    if confirmation:
        steps_for_retrieval += [f"Confirm: {c}" for c in confirmation]
    if do_not:
        steps_for_retrieval += [f"Do not: {d}" for d in do_not]
    if common_sources:
        steps_for_retrieval += [f"Common source: {s}" for s in common_sources]

    return "\n".join([
        f"Title: {title}",
        f"Category: {category}",
        f"Severity: {severity}",
        f"Applies to years: {ymin}–{ymax}",
        f"Customer summary: {description}",
        "Symptoms: " + " | ".join(symptoms),
        "Clarifying questions: " + " | ".join(clarifying_questions),
        "Steps: " + " | ".join(steps_for_retrieval),
        "Model year notes: " + " | ".join(model_year_notes),
        "Stop & escalate: " + " | ".join(stop_and_escalate),
        f"Next step: {recommended_action}",
    ])


def main():
    if not os.path.exists(ARTICLES_FILE):
        raise FileNotFoundError(f"Could not find {ARTICLES_FILE}")

    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your file shape:
    # { "knowledge_base": { "scope": {...}, "articles": [ ... ] } }
    if isinstance(data, dict) and "knowledge_base" in data and isinstance(data["knowledge_base"], dict):
        kb = data["knowledge_base"]
        articles = kb.get("articles", None)
        if not isinstance(articles, list):
            raise ValueError(f"'knowledge_base.articles' must be a list. Found: {type(articles)}")
    elif isinstance(data, dict) and "articles" in data and isinstance(data["articles"], list):
        articles = data["articles"]
    elif isinstance(data, list):
        articles = data
    else:
        raise ValueError(
            "vinnies_brain_articles.json must be a list of articles, or contain 'articles', "
            "or contain 'knowledge_base.articles'."
        )

    # Normalize any string-encoded articles
    articles = [normalize_article(a) for a in articles]

    # Validate objects
    for i, a in enumerate(articles):
        if not isinstance(a, dict):
            raise ValueError(f"Article at index {i} is not an object. Type={type(a)} Value={str(a)[:200]}")

    print(f"Loaded {len(articles)} articles")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()

        sql = """
        INSERT INTO kb_articles (
            id,
            title,
            category,
            severity,
            years_min,
            years_max,
            customer_summary,
            clarifying_questions,
            steps,
            model_year_notes,
            stop_and_escalate,
            next_step,
            retrieval_text,
            embedding
        )
        VALUES (
            %(id)s,
            %(title)s,
            %(category)s,
            %(severity)s,
            %(years_min)s,
            %(years_max)s,
            %(customer_summary)s,
            %(clarifying_questions)s::jsonb,
            %(steps)s::jsonb,
            %(model_year_notes)s::jsonb,
            %(stop_and_escalate)s::jsonb,
            %(next_step)s,
            %(retrieval_text)s,
            %(embedding)s::vector
        )
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            category = EXCLUDED.category,
            severity = EXCLUDED.severity,
            years_min = EXCLUDED.years_min,
            years_max = EXCLUDED.years_max,
            customer_summary = EXCLUDED.customer_summary,
            clarifying_questions = EXCLUDED.clarifying_questions,
            steps = EXCLUDED.steps,
            model_year_notes = EXCLUDED.model_year_notes,
            stop_and_escalate = EXCLUDED.stop_and_escalate,
            next_step = EXCLUDED.next_step,
            retrieval_text = EXCLUDED.retrieval_text,
            embedding = EXCLUDED.embedding,
            updated_at = now();
        """

        rows = []

        for i, article in enumerate(articles, start=1):
            title = article.get("title", f"(untitled {i})")
            print(f"Embedding article {i}/{len(articles)}: {title}")

            ymin, ymax = get_year_range(article)

            # Map your schema fields into DB columns
            description = article.get("description", "")

            # Store symptoms as clarifying_questions (still useful for UI prompts)
            clarifying_questions_list = norm_list(article.get("symptoms", []))

            # Store initial_checks as steps
            steps_list = norm_list(article.get("initial_checks", []))

            model_year_notes_list = model_year_notes_to_list(article.get("model_year_notes", {}))

            stop_list = norm_list(article.get("stop_and_escalate", article.get("escalate_if", [])))

            next_step = article.get("recommended_action", "")

            retrieval_text = make_retrieval_text(article)
            embedding = embed_text(retrieval_text)
            embedding_lit = to_vector_literal(embedding)

            rows.append({
                "id": article["id"],
                "title": article.get("title", ""),
                "category": article.get("category", ""),
                "severity": article.get("severity", ""),
                "years_min": ymin,
                "years_max": ymax,
                "customer_summary": description,
                "clarifying_questions": json.dumps(clarifying_questions_list),
                "steps": json.dumps(steps_list),
                "model_year_notes": json.dumps(model_year_notes_list),
                "stop_and_escalate": json.dumps(stop_list),
                "next_step": next_step,
                "retrieval_text": retrieval_text,
                "embedding": embedding_lit,
            })

        execute_batch(cur, sql, rows, page_size=5)
        conn.commit()

        print("✅ Knowledge base seeded successfully")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
