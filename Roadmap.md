# Roadmap

Tracking status against the [project audit](PROJECT_AUDIT.md).

## Done

- Removed the stray copy of the project template (`README.md`, `ETHICS.md`,
  `requirements.txt`, `starter_code.py`, `starter_notebook.ipynb`) from
  `embeddings/`. That folder now only holds downloaded GloVe files.
- Clarified in `README.md` that `starter_code.py` is the instructor/reference
  implementation and `starter_notebook.ipynb` is a separate, intentionally
  blank student worksheet — the two are not meant to be kept in sync.
- Fixed the `requirements.txt` comment on `imbalanced-learn`: it's not
  imported anywhere in `starter_code.py` (`compute_class_weights` uses
  `sklearn.utils.class_weight`), so it's noted as optional for experimenting
  with resampling, not as part of the required class-weighting path.

## Not done

- **End-to-end run against real data.** Nobody has downloaded the AllSides
  CSV or GloVe 300d embeddings and run `python starter_code.py` start to
  finish. The "completed" implementation is unverified against real data —
  do this before trusting the accuracy claims in the README.
- Fix whatever breaks during that run (data loading edge cases, memory
  limits, etc.) and record actual achieved accuracy here once known.
