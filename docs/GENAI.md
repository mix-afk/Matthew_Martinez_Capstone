# Use of Generative AI (Step 9)

Generative AI was used in two ways: as a feature inside the product, and as a helper while building the project.

## 1. In the product: "why this item" lines

Every recommendation in the API and demo page carries a one-line reason, for example "A new color of something you already love".

How it works (`src/recsys/genai.py`):

1. **Facts first.** For each recommended item, code turns the ranker's own features into plain facts: bought this before, another color of a product the customer bought, often bought together with an item in their history, matches the product types or colors they buy most, sales rising this week, one of this week's best sellers.
2. **LLM writes the line.** When `ANTHROPIC_API_KEY` is set and the caller asks for `explain=llm`, the facts go to Claude (`claude-haiku-4-5-20251001` by default, set with `RECSYS_LLM_MODEL`) with a short prompt: use only the given facts, max 16 words, never mention age, gender or any personal trait.
3. **Guardrail.** The reply is checked for personal-trait words and length. If it fails, or the API errors, the app falls back to a deterministic template for the first fact. The response says which one was used (`reason_source: llm | template | template_guardrail`).
4. **No key, no problem.** Without a key everything runs on templates, so tests, CI and the demo never depend on an external service.

Why this design:

- The LLM never decides *what* to recommend. The ranker does. The LLM only phrases a reason from facts the ranker actually used, so it cannot make up a reason that is not true.
- Age is used by the ranker (see the fairness section of the report), but the reason text is never allowed to mention it.
- Cost stays low: reasons are short, and the batch job can pre-generate them once per customer per week.

Examples are in `notebooks/05_genai_and_serving.ipynb` (cells 2 to 5) and in the demo GIF.

## 2. LLM-assisted documentation

`genai.column_profile()` builds a compact profile of a table (type, missing %, unique count, range or top values per column) and `genai.draft_data_dictionary()` sends it to Claude with a prompt that asks for a description, type and allowed values per column, and to write "unknown" when the profile does not support a meaning.

The data dictionary in `docs/DATA_DICTIONARY.md` started from this kind of draft. I then checked every line against the data (row counts, ranges and category counts were recomputed on the full files, see notebook 01) and corrected it by hand. The EDA summaries in the report were drafted the same way from notebook outputs and then edited and checked against the numbers.

## 3. AI assistance while building

I used Claude as a coding assistant for scaffolding the repository, writing first drafts of modules and tests, and debugging (for example, it caught that LightGBM early stopping was tracking log loss instead of AUC). All design choices, results and conclusions were reviewed by me, and every number in the report comes from the notebooks in this repository.

## 4. Demo

`demo/demo.gif` shows the storefront page with reasons on each card. With a key set, tick "AI reasons" on the page to switch to LLM-written lines.
