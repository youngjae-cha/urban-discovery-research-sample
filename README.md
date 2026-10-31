# Urban discovery: selected research code and results

**Selected research sample from an ongoing, unpublished study.** This sample contains selected code and results from an ongoing research project. The data behind the study and the complete research pipeline remain private.

## Research question

Where do everyday urban places offer opportunities for exploration, discovery and learning, and how does their spatial availability relate to cities' intellectual and social life?

The project analyzed 2019 Google Maps reviews of **382,361 places across 387 U.S. metropolitan areas and 34 broad venue types**. Review-based place scores identify discovery places relative to other places of the same type. Discovery coverage is the share of included places within 1 km of another discovery place in the same metropolitan area.

## Selected results

![Actual metropolitan patterns](figures/01_actual_city_patterns.png)

Each dot represents one of the 387 metropolitan areas. The left panel compares the share of discovery places with discovery coverage; the right panel shows the coverage distribution. This is an unchanged static figure based on actual study aggregates. Its metropolitan-level source table is withheld, so this figure cannot be regenerated from the sample.

![Saved associations and weighting sensitivity](figures/02_saved_associations.png)

The left panel shows four previously estimated, adjusted associations, with stored 95% confidence intervals. The attitude measure is a **sexual-orientation IAT**, with its sign reversed; it is not a measure of bias toward every minority group. The climate measure is an opinion composite, and innovation is represented by patent activity.

The right panel retains a consequential sensitivity check: the learning estimate is smaller and its interval includes zero when national respondent weights are used to aggregate the survey responses. These are respondent-aggregation choices, not weights on cities in the regression. The figures describe associations and measurement sensitivity; they do not establish causal effects.

## Representative code

- `code/text_normalization_excerpt.py`: the unchanged normalization function used in the review pipeline, including Unicode normalization, punctuation handling and URL/email placeholders. The private context dictionaries and classifiers are omitted.
- `code/spatial_coverage_excerpt.py`: unchanged original functions for within-type midpoint ranking, a 90th-percentile cutoff, same-metropolitan-area haversine distance, self exclusion, and direct distance checks. Legacy `2018` output-column names are retained from the original audit; they do not set the input year.
- `code/estimation_excerpt.py`: unchanged original functions for within-sample standardization, HC1 uncertainty, t-based confidence intervals and full/reduced models on identical complete cases.
- `code/result_figures.py`: the display functions used to redraw the selected saved estimates.

The representative functions are available for inspection. They require authorized analysis frames, which are **not included**, and are not invoked on private data by the notebook. The text classifiers, semantic directions, filtering dictionaries and trained parameters are withheld. This sample does not classify reviews or reconstruct place scores.

## Executed notebook and running it

Open [`notebooks/SELECTED_RESULTS_EXECUTED.ipynb`](notebooks/SELECTED_RESULTS_EXECUTED.ipynb) to inspect completed cells and embedded figures. It reads only the two included summary CSVs, redraws the estimate figure and displays the static city figure. It does not refit regressions, train models or access external services.

[Open the GitHub-rendered notebook directly](https://notebooks.githubusercontent.com/view/ipynb?browser=unknown_browser&bypass_fastly=true&color_mode=auto&commit=2a56b9fc9a12573722ce71195446b4ad44e2c121&device=unknown_device&docs_host=https%3A%2F%2Fdocs.github.com&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f796f756e676a61652d6368612f757262616e2d646973636f766572792d72657365617263682d73616d706c652f326135366239666339613132353733373232636537313139353434366234616434346532633132312f6e6f7465626f6f6b732f53454c45435445445f524553554c54535f45584543555445442e6970796e62&link_underline_enabled=true&logged_in=true&nwo=youngjae-cha%2Furban-discovery-research-sample&path=notebooks%2FSELECTED_RESULTS_EXECUTED.ipynb&platform=unknown_platform&repository_id=1440648804&repository_type=Repository&version=0#a78fede8-3280-4129-bbc4-6ec74c8ea859) if the embedded GitHub preview stalls. This view shows the same executed notebook, pinned to its original publication commit.

Tested with Python 3.13:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/execute.py
```

Re-execution refreshes the notebook, writes the estimate figure under `figures/`, and places an execution receipt under ignored `outputs/`.

## Included and withheld material

`results/` contains **four model-level estimates and two respondent-aggregation specifications**, limited to their labels, estimates, intervals and sample sizes. It contains no metropolitan analysis rows, individual reviews, reviewer identifiers, coordinates, survey responses or trained-model files. No private prompts, manuscripts, API credentials or institutional paths are included.

`PUBLIC_FILES.json` is the closed list of proposed public files. This is a selected research sample, not a complete replication package. No new license or rights in the underlying data are granted by this sample.
