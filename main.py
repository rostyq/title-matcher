# /// script
# dependencies = [
#     "marimo>=0.25.2",
#     "numpy==2.5.5",
#     "polars==2.1.0",
#     "rapidfuzz==3.14.6",
#     "scikit-learn==1.9.1",
# ]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", sql_output="lazy-polars")

with app.setup:
    import re
    from functools import partial
    from string import punctuation

    import marimo as mo
    import numpy as np
    import polars as pl
    from rapidfuzz import fuzz, process
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer


@app.cell(hide_code=True)
def title():
    mo.md(r"""
    # Title matcher

    Match noisy, human-typed titles (typos, Cyrillic, duplicates) against a clean vocabulary.

    Each input below takes **either** an uploaded file **or** text pasted from the clipboard
    (e.g. a column copied from Excel). If both are empty, the sample file from `data/` is used.
    """)
    return


@app.cell
def helpers():
    DATA_DIR = mo.notebook_dir() / "data"


    def decode_bytes(raw: bytes) -> str:
        """Decode an uploaded text file: UTF-8 / UTF-16 (Excel "Unicode text") / cp1251."""
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("cp1251")


    def parse_titles(text: str) -> list[str]:
        """Each line is a title. Whitespace is collapsed, blank lines dropped, duplicates kept."""
        return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


    def resolve_source(file_ui, text_ui, sample_name: str, use_sample: bool):
        """Pick file > pasted text > sample file. Returns (titles, source label)."""
        if file_ui.value:
            f = file_ui.value[0]
            return parse_titles(decode_bytes(f.contents)), f"file: {f.name}"
        if text_ui.value.strip():
            return parse_titles(text_ui.value), "pasted text"
        if use_sample:
            path = DATA_DIR / sample_name
            return parse_titles(decode_bytes(path.read_bytes())), f"sample: data/{sample_name}"
        return [], "empty"

    return (resolve_source,)


@app.cell
def input_widgets():
    _filetypes = [".txt"]
    vocab_file = mo.ui.file(filetypes=_filetypes, label="Upload vocabulary")
    vocab_text = mo.ui.text_area(placeholder="…or paste vocabulary titles, one per line", rows=8, full_width=True)
    input_file = mo.ui.file(filetypes=_filetypes, label="Upload input titles")
    input_text = mo.ui.text_area(placeholder="…or paste input titles, one per line", rows=8, full_width=True)
    stop_file = mo.ui.file(filetypes=_filetypes, label="Upload stop words")
    stop_text = mo.ui.text_area(placeholder="…or paste stop words, one per line (optional)", rows=8, full_width=True)
    use_samples = mo.ui.checkbox(value=True, label="Fall back to sample files from `data/` when an input is empty")
    return (
        input_file,
        input_text,
        stop_file,
        stop_text,
        use_samples,
        vocab_file,
        vocab_text,
    )


@app.cell(hide_code=True)
def input_layout(
    input_file,
    input_text,
    stop_file,
    stop_text,
    use_samples,
    vocab_file,
    vocab_text,
):
    mo.vstack(
        [
            mo.md("## Inputs"),
            mo.hstack(
                [
                    mo.vstack([mo.md("### Vocabulary"), vocab_file, vocab_text]),
                    mo.vstack([mo.md("### Input titles"), input_file, input_text]),
                    mo.vstack([mo.md("### Stop words *(optional)*"), stop_file, stop_text]),
                ],
                widths="equal",
                gap=1.5,
            ),
            use_samples,
        ]
    )
    return


@app.cell
def resolve_inputs(
    input_file,
    input_text,
    resolve_source,
    stop_file,
    stop_text,
    use_samples,
    vocab_file,
    vocab_text,
):
    vocabulary, vocab_source = resolve_source(vocab_file, vocab_text, "vocabulary.txt", use_samples.value)
    raw_inputs, input_source = resolve_source(input_file, input_text, "input.txt", use_samples.value)
    _stop, stop_source = resolve_source(stop_file, stop_text, "stopwords.txt", use_samples.value)
    stopwords = sorted({w.lower() for w in _stop})
    return (
        input_source,
        raw_inputs,
        stop_source,
        stopwords,
        vocab_source,
        vocabulary,
    )


@app.cell
def input_summary(
    input_source,
    raw_inputs,
    stop_source,
    stopwords,
    vocab_source,
    vocabulary,
):
    def _card(title, items, source):
        _unique = len(dict.fromkeys(items))
        return mo.vstack(
            [
                mo.stat(value=len(items), label=title, caption=f"{_unique} unique · {source}", bordered=True),
                mo.ui.table([{title: t} for t in items], page_size=8, selection=None),
            ]
        )


    mo.accordion(
        {
            "Parsed inputs preview": mo.hstack(
                [
                    _card("Vocabulary", vocabulary, vocab_source),
                    _card("Input titles", raw_inputs, input_source),
                    _card("Stop words", stopwords, stop_source),
                ],
                widths="equal",
                gap=1.5,
            )
        }
    )
    return


@app.cell(hide_code=True)
def matching_md():
    mo.vstack(
        [
            mo.md("## Matching"),
            mo.accordion(
                {
                    "How matching works": mo.md(
                        r"""
                        Every title is **normalized** (Cyrillic → Latin transliteration, lowercase, punctuation removed,
                        `+` → `plus`), then **tokenized** (split on spaces/hyphens, stop words dropped, letters and
                        digits separated: `s24` → `s 24`).

                        Two matchers score every input against every vocabulary title:

                        - **TF‑IDF**: an sklearn pipeline over words, word bigrams and character trigrams, compared with cosine similarity;
                        - **rapidfuzz**: an edit-distance score on the normalized token strings.

                        For each input, the matcher with the higher score wins. Scores below the threshold are reported as *no match*.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def text_processing():
    LATIN_TABLE = str.maketrans(
        "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЮЯЬабвгґдеєжзиіїйклмнопрстуфхцчшщюяьЫыЭэЁё",
        "ABVHGDEÊŽZYIÏJKLMNOPRSTUFXCČŠŜÛÂʹabvhgdeêžzyiïjklmnoprstufxcčšŝûâʹYyEeÊê",
        (punctuation + "«»Ъъ").translate(str.maketrans("", "", "+ʹ-_.")),
    )

    _WORD_SPLIT = re.compile(r"\s+|-+")
    _DIGIT_SPLIT = re.compile(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)")


    def normalize(s: str) -> str:
        s = s.translate(LATIN_TABLE).lower()
        return s.replace("ʹ", "").replace("+", " plus ")


    def normalize_all(titles) -> list[str]:
        return [normalize(s) for s in titles]


    def tokenize(s: str, stop_words: frozenset[str] = frozenset()) -> list[str]:
        """Split normalized text into words, drop stop words, then split letter/digit runs."""
        return [
            token
            for word in _WORD_SPLIT.split(s)
            if word and word not in stop_words
            for token in _DIGIT_SPLIT.split(word)
            if token
        ]


    def ngrams(seq, n: int):
        return zip(*(seq[i:] for i in range(n)))


    def analyze(s: str, stop_words: frozenset[str] = frozenset()) -> list[str]:
        """Features of a normalized title: words + word bigrams + per-word char trigrams."""
        words = tokenize(s, stop_words)
        word_bigrams = [" ".join(g) for g in ngrams(words, 2)]
        char_trigrams = ["".join(g) for word in words for g in ngrams(word, 3)]
        return words + word_bigrams + char_trigrams

    return analyze, normalize, normalize_all, tokenize


@app.cell
def stop_word_set(normalize, stopwords):
    STOP_WORDS = frozenset(w for w in (normalize(s).strip() for s in stopwords) if w)
    return (STOP_WORDS,)


@app.function
def find_best_matches(names: list[str], choices: list[str], sim_matrix: np.ndarray) -> pl.DataFrame:
    """Best-scoring choice for every name (row-wise argmax of the similarity matrix)."""
    best_idx = np.argmax(sim_matrix, axis=1)
    best_scores = sim_matrix[np.arange(len(names)), best_idx]
    return pl.DataFrame(
        {
            "name": names,
            "best_match": [choices[i] for i in best_idx],
            "similarity_score": best_scores.tolist(),
        }
    )


@app.cell
def match_controls():
    fuzz_scorer = mo.ui.dropdown(
        options={
            "token_sort_ratio": fuzz.token_sort_ratio,
            "ratio": fuzz.ratio,
            "QRatio": fuzz.QRatio,
            "token_set_ratio": fuzz.token_set_ratio,
            "WRatio": fuzz.WRatio,
        },
        value="token_sort_ratio",
        label="rapidfuzz scorer",
    )
    match_threshold = mo.ui.slider(0.0, 1.0, step=0.01, value=0.5, label="Match threshold", show_value=True)
    top_k = mo.ui.number(start=1, stop=10, step=1, value=3, label="Top-k candidates")
    mo.hstack([fuzz_scorer, match_threshold, top_k], justify="start", gap=2)
    return fuzz_scorer, match_threshold, top_k


@app.cell
def sklearn_model(STOP_WORDS, analyze, normalize_all, raw_inputs, vocabulary):
    title_model = Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_all)),
            ("vectorize", TfidfVectorizer(analyzer=partial(analyze, stop_words=STOP_WORDS))),
        ]
    )
    _vocab_vectors = title_model.fit_transform(vocabulary)
    tfidf_sim = cosine_similarity(title_model.transform(raw_inputs), _vocab_vectors)
    return (tfidf_sim,)


@app.cell
def rapidfuzz_scores(
    STOP_WORDS,
    fuzz_scorer,
    normalize,
    raw_inputs,
    tokenize,
    vocabulary,
):
    def _prepare(s: str) -> str:
        return " ".join(tokenize(normalize(s), STOP_WORDS))


    fuzz_sim = (
        process.cdist(
            [_prepare(s) for s in raw_inputs],
            [_prepare(s) for s in vocabulary],
            scorer=fuzz_scorer.value,
            workers=-1,
        )
        / 100
    )
    return (fuzz_sim,)


@app.cell
def combine_matches(
    STOP_WORDS,
    fuzz_sim,
    match_threshold,
    normalize,
    raw_inputs,
    tfidf_sim,
    tokenize,
    vocabulary,
):
    _tfidf = find_best_matches(raw_inputs, vocabulary, tfidf_sim)
    _fuzz = find_best_matches(raw_inputs, vocabulary, fuzz_sim)

    matches = (
        pl.DataFrame({"input": raw_inputs, "tokens": [" ".join(tokenize(normalize(s), STOP_WORDS)) for s in raw_inputs]})
        .with_columns(
            tfidf_match=_tfidf["best_match"],
            tfidf_score=_tfidf["similarity_score"],
            fuzz_match=_fuzz["best_match"],
            fuzz_score=_fuzz["similarity_score"],
        )
        .with_columns(
            method=pl.when(pl.col("tfidf_score") >= pl.col("fuzz_score")).then(pl.lit("tfidf")).otherwise(pl.lit("rapidfuzz")),
            score=pl.max_horizontal("tfidf_score", "fuzz_score"),
        )
        .with_columns(
            match=pl.when(pl.col("score") < match_threshold.value)
            .then(None)
            .when(pl.col("method") == "tfidf")
            .then(pl.col("tfidf_match"))
            .otherwise(pl.col("fuzz_match")),
            agree=pl.col("tfidf_match") == pl.col("fuzz_match"),
            score_gap=(pl.col("tfidf_score") - pl.col("fuzz_score")).abs(),
        )
        .select("input", "match", "score", "method", "agree", "score_gap", "tokens", "tfidf_match", "tfidf_score", "fuzz_match", "fuzz_score")
    )
    return (matches,)


@app.cell(hide_code=True)
def match_stats(match_threshold, matches):
    _matched = matches.filter(pl.col("match").is_not_null())
    mo.hstack(
        [
            mo.stat(value=matches.height, label="Input titles", bordered=True),
            mo.stat(value=_matched.height, label="Matched", caption=f"{_matched['match'].n_unique()} distinct vocabulary titles", bordered=True),
            mo.stat(value=matches.height - _matched.height, label="No match", caption=f"score < {match_threshold.value:.2f}", bordered=True),
            mo.stat(value=int((~matches["agree"]).sum()), label="Matchers disagree", caption="worth reviewing", bordered=True),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def results_tabs(copy_view, match_threshold, match_view, method_view, top_k):
    _matches_help = mo.md(
        f"""
        - **input**: the title as it was provided.
        - **match**: the vocabulary title chosen by whichever matcher (TF‑IDF or rapidfuzz) scored higher;
          empty when the best score is below the threshold ({match_threshold.value:.2f}).
        - **tfidf_top_k**: the next {int(top_k.value)} closest vocabulary titles by TF‑IDF similarity, best first.
        - **fuzz_top_k**: the next {int(top_k.value)} closest vocabulary titles by rapidfuzz similarity, best first.

        Both top‑k columns leave out the chosen **match**, so they list *alternatives* to review.
        If the two matchers disagreed, the losing matcher's own best guess appears first in its column.
        Change *k* with the **Top-k candidates** control above.
        """
    )

    mo.ui.tabs(
        {
            "Matches": mo.vstack(
                [
                    mo.accordion({"What do these columns mean?": _matches_help}),
                    mo.ui.table(match_view, selection=None, page_size=20),
                ]
            ),
            "Copy columns": copy_view,
            "TF-IDF vs rapidfuzz": mo.ui.table(method_view, selection=None, page_size=20),
        }
    )
    return


@app.cell(hide_code=True)
def match_view(fuzz_sim, matches, tfidf_sim, top_k, vocabulary):
    def _top_k_alternatives(sim: np.ndarray, chosen: list[str | None], k: int) -> list[str]:
        """Top-k vocabulary titles per input, excluding the chosen match, joined with ' | '."""
        _idx = np.argsort(-sim, axis=1)[:, : k + 1]
        return [
            " | ".join([vocabulary[j] for j in idx if vocabulary[j] != match][:k])
            for idx, match in zip(_idx, chosen)
        ]


    _chosen = matches["match"].to_list()
    _k = int(top_k.value)
    match_view = matches.select("input", "match").with_columns(
        tfidf_top_k=pl.Series(_top_k_alternatives(tfidf_sim, _chosen, _k), dtype=pl.String),
        fuzz_top_k=pl.Series(_top_k_alternatives(fuzz_sim, _chosen, _k), dtype=pl.String),
    )
    return (match_view,)


@app.cell(hide_code=True)
def copy_view(matches):
    def _copy_block(column: str):
        # One line per input row; no match -> empty line, so rows stay aligned when pasted into Excel.
        _lines = "\n".join(v or "" for v in matches[column].to_list())
        return mo.vstack(
            [
                mo.md(f"**{column}**"),
                mo.md(f"```text\n{_lines}\n```").style({"max-height": "420px", "overflow": "auto"}),
            ]
        )


    copy_view = mo.vstack(
        [
            mo.md(f"{matches.height} lines per column, aligned with the input (no match → empty line). Hover a block and click the copy icon."),
            mo.hstack([_copy_block(c) for c in ("match", "tfidf_match", "fuzz_match")], widths="equal", gap=1),
        ]
    )
    return (copy_view,)


@app.cell(hide_code=True)
def method_view(matches):
    method_view = matches.select("input", "tfidf_match", "tfidf_score", "fuzz_match", "fuzz_score")
    return (method_view,)


@app.cell(hide_code=True)
def disagree_explorer(matches):
    disagree_table = mo.ui.table(
        matches.with_row_index("row")
        .filter(~pl.col("agree"))
        .select("row", "input", "match", "method", "tfidf_match", "tfidf_score", "fuzz_match", "fuzz_score"),
        selection="single",
        page_size=10,
        label="Inputs where TF-IDF and rapidfuzz picked different titles. Select a row to inspect it.",
    )
    mo.vstack([mo.md("## Disagreements"), disagree_table])
    return (disagree_table,)


@app.cell(hide_code=True)
def disagree_detail(
    STOP_WORDS,
    disagree_table,
    fuzz_sim,
    normalize,
    tfidf_sim,
    tokenize,
    vocabulary,
):
    def _top_k(scores: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        return [(vocabulary[i], float(scores[i])) for i in np.argsort(-scores)[:k]]


    def _tokens(s: str) -> str:
        return " · ".join(tokenize(normalize(s), STOP_WORDS))


    if disagree_table.data.is_empty():
        _view = mo.callout(mo.md("Both matchers agree on every input."), kind="success")
    else:
        _selected = disagree_table.value
        _row = (_selected if not _selected.is_empty() else disagree_table.data).row(0, named=True)
        _i = _row["row"]
        _tf, _rf = _top_k(tfidf_sim[_i]), _top_k(fuzz_sim[_i])
        _candidates = pl.DataFrame(
            {
                "rank": range(1, len(_tf) + 1),
                "tfidf_candidate": [c for c, _ in _tf],
                "tfidf_score": [s for _, s in _tf],
                "tfidf_candidate_tokens": [_tokens(c) for c, _ in _tf],
                "fuzz_candidate": [c for c, _ in _rf],
                "fuzz_score": [s for _, s in _rf],
                "fuzz_candidate_tokens": [_tokens(c) for c, _ in _rf],
            }
        )
        _chosen = _row["match"] or "*no match (below threshold)*"
        _view = mo.vstack(
            [
                mo.md(
                    f"""
                    ### `{_row["input"]}`

                    **Tokens:** `{_tokens(_row["input"])}`
                    **Chosen:** {_chosen} (by **{_row["method"]}**)
                    """
                ),
                mo.ui.table(_candidates, selection=None, pagination=False),
            ]
        )
    _view
    return


if __name__ == "__main__":
    app.run()
