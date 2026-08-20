# Project Audit — Political Bias Detection Workshop

_Audit date: 2026-08-20_

## Where the project stands

This is workshop/educational material (not a production system) for building a
political bias classifier (Left/Center/Right) on AllSides news articles, using
a stacked bidirectional LSTM with GloVe 300d embeddings. `ETHICS.md` and
`README.md` are thorough and appropriate for a workshop setting.

### Implementation status

- **`starter_code.py` (root, 904 lines):** fully implemented. All 14
  functions — `load_data`, `preprocess_text`, `compute_class_weights`,
  `load_glove_embeddings`, `create_embedding_matrix`, `build_model`,
  `train_model`, `evaluate_model`, `analyze_predictions`, `predict_bias`,
  `test_on_real_articles`, `main()`, etc. — have real code, not stubs.
  Remaining `TODO` comments are instructional hints for students, not
  unimplemented logic.
- **`starter_notebook.ipynb`:** still a pure stub. 11 TODO cells, 0/13 code
  cells executed. This is the student-facing worksheet and has not been
  synced with the completed `starter_code.py`.
- **`data/`:** empty aside from a README with download instructions. No
  `allsides_news_complete.csv` present.
- **`embeddings/`:** no GloVe file present. It does contain a stray,
  duplicate copy of `README.md`, `ETHICS.md`, `requirements.txt`,
  `starter_notebook.ipynb`, and an *older, less-complete* version of
  `starter_code.py` (86 TODOs vs. 0 unimplemented in the root version).
  This looks like an accidental copy of the whole template into the wrong
  folder rather than intentional content.
- **`Roadmap.md`:** effectively a placeholder — one line, "This should fix
  some problems."

## Next steps

1. **Clean up `embeddings/`.** Remove the 5 stray files (`README.md`,
   `ETHICS.md`, `requirements.txt`, `starter_code.py`, `starter_notebook.ipynb`)
   that don't belong there. The folder should hold only the downloaded GloVe
   embeddings (or a `.gitkeep` placeholder until then).

2. **Decide the notebook's role and act on it.**
   - If `starter_notebook.ipynb` is meant to be the "answer key" mirroring
     `starter_code.py`, port the completed implementations into its TODO
     cells and execute it end-to-end so outputs are saved.
   - If it's meant to stay a blank student exercise, leave it as-is but say
     so explicitly in `README.md` so it's clear the notebook and the root
     `starter_code.py` serve different audiences (student vs. instructor/
     reference).

3. **Write real content in `Roadmap.md`**, or delete it if it's not adding
   value. Right now it's a placeholder sentence with no actionable
   information.

4. **Get end-to-end verification.** Nothing in the pipeline has actually
   been run:
   - Download the AllSides dataset into `data/allsides_news_complete.csv`.
   - Download GloVe 300d embeddings into `embeddings/glove.840B.300d.txt`.
   - Run `python starter_code.py` and confirm the pipeline completes,
     producing a model in the 65–80% accuracy range described in the README.
   - Fix anything that breaks — the "finished" implementation has not been
     smoke-tested against real data.

5. **Minor: `requirements.txt` inconsistency.** `imbalanced-learn` is listed
   as "optional but recommended" in a comment, but the README/ETHICS docs
   treat class-weighting as essential. Either drop the optional framing or
   make clear it's not required since `compute_class_weights` uses
   `sklearn.utils.class_weight` directly, not `imbalanced-learn`.
