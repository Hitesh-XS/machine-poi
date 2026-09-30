# Literature map data

Data behind [research directions](../research_directions.md), retrieved from
arXiv and Semantic Scholar on 2026-09-30. Citation counts move; treat them as a
snapshot.

| File | Contents |
| --- | --- |
| [`focus_corpus.csv`](focus_corpus.csv) | The 732 focus papers, ranked by relevance to this repository's docs |
| [`reading_list.md`](reading_list.md) | Uncited high-relevance work, nearest papers per strand and bridges |
| [`theme_growth.csv`](theme_growth.csv) | Papers per year for each of the 12 queries, over the full 7,322-paper harvest |
| [`cluster_growth.csv`](cluster_growth.csv) | Papers per year for each focus-corpus cluster |
| [`theme_growth.png`](theme_growth.png) | Growth chart used in the directions page |
| [`queries.json`](queries.json) | The arXiv queries, date range and method settings |

Columns in `focus_corpus.csv`:

| Column | Meaning |
| --- | --- |
| `rank`, `relevance` | TF-IDF relevance to the repository's docs, with extra weight on core terms |
| `arxiv_id`, `title`, `published`, `authors`, `primary_category` | arXiv metadata; authors are truncated to three |
| `cluster` | One of 12 k-means clusters, labelled by hand |
| `citations`, `citations_per_month` | Semantic Scholar counts; blank when it had no record (110 papers) |
| `cited_by_project` | The paper is mentioned in this repository's docs |
| `sim_A_steering` … `sim_D_containment` | Cosine similarity to each strand's documents |
| `top40_strands`, `top40_strand_count` | Strands for which the paper is in the 40 most similar |

2026 counts in the growth tables cover January to September; the
`2026_annualized` column divides them by 0.75. Growth ratios on small 2024
counts are noisy.
