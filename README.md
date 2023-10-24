# CausalBench
![Python version](https://img.shields.io/badge/Python-3.8-blue)
![Library version](https://img.shields.io/badge/Version-1.1.1-blue)

## Introduction

Mapping biological mechanisms in cellular systems is a fundamental step in early stage drug discovery that serves to generate hypotheses on what disease-relevant molecular targets may effectively be modulated by pharmacological interventions. With the advent of high-throughput methods for measuring single-cell gene expression under genetic perturbations, we now have effective means for generating evidence for causal gene-gene interactions at scale. However, inferring graphical networks of the size typically encountered in real-world gene-gene interaction networks is difficult in terms of both achieving and evaluating faithfulness to the true underlying causal graph. Moreover, standardised benchmarks for comparing methods for causal discovery in perturbational single-cell data do not yet exist. Here, we introduce CausalBench - a comprehensive benchmark suite for evaluating network inference methods on large-scale perturbational single-cell gene expression data. CausalBench introduces several biologically meaningful performance metrics and operates on two large, curated and openly available benchmark data sets for evaluating methods on the inference of gene regulatory networks from single-cell data generated under perturbations. With real-world datasets consisting of over 200 000 training samples under interventions, CausalBench could potentially help facilitate advances in causal network inference by providing what is - to the best of our knowledge - the largest openly available test bed for causal discovery from real-world perturbation data to date.

## CausalBench ICLR-23 Challenge

Learn more about the CausalBench challenge for graph inference on perturbational gene expression data [here](https://www.gsk.ai/causalbench-challenge/).

## Datasets

- RPE1 day 7 Perturb-seq (RD7): targeting DepMap essential genes at day 7 after transduction
- K562 day 6 Perturb-seq (KD7): targeting DepMap essential genes at day 6 after transduction


## Training Regimes

- Observational: only observational data is given as training data to the model.
- Observational and partial interventional: observational as well as interventional data for part of the variables is given as training data to the model.
- Observational and full interventional: observational as well as interventional data for all the variables is given as training data to the model.

## Install

```bash
pip install causalbench
```

## How to run the benchmark?

Example of command to run a model on the k562 dataset in the observational regime. 

```bash
causalbench_run \
    --dataset_name weissmann_k562 \
    --output_directory /path/to/output/ \
    --data_directory /path/to/data/storage \
    --training_regime observational \
    --model_name pc \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter \
    --max_path_length -1 \
    --omission_estimation_size 500
```

Results are written to the folder at `/path/to/output/`, and processed datasets will be cached at `/path/to/data/storage`. See the MainApp class for more hyperparameter options, especially in the (partial) interventional setting.

## Add a model

New models can be easily added. The only contract for a model is to implement the [AbstractInferenceModel] class.

```python
from causalscbench.models.abstract_model import AbstractInferenceModel

class FullyConnected(AbstractInferenceModel):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        random.seed(seed)
        edges = set()
        for i in range(len(gene_names)):
            a = gene_names[i]
            for j in range(i + 1, len(gene_names)):
                b = gene_names[j]
                edges.add((a, b))
                edges.add((b, a))
        return list(edges)
```

## Citation

Please consider citing, if you reference or use our methodology, code or results in your work: 

    @article{chevalley2022causalbench,
        title={{CausalBench: A Large-scale Benchmark for Network Inference from Single-cell Perturbation Data}},
        author={Chevalley, Mathieu and Roohani, Yusuf and Mehrjou, Arash and Leskovec, Jure and Schwab, Patrick},
        journal={arXiv preprint arXiv:2210.17283},
        year={2022}
    }


### License

[License](LICENSE.txt)

## External data

Data in the /data_access/data folder were aggregated from the following resource: "ChIP-Atlas © Shinya Oki (Kyoto University) licensed under CC Attribution-Share Alike 4.0 International" [link to license](https://dbarchive.biosciencedbc.jp/en/chip-atlas/lic.html). The adapted datasets based on a part of the original database is here redistributed under the same license, which can be found in the LICENSE.txt file.

This codebase also links to muliple data sources to be downloaded.  The associated licenses are summarized here:

Replogle et al (perturb-seq screen): https://gwps.wi.mit.edu/ (LICENSE: CC-BY-4.0)

CORUM: http://mips.helmholtz-muenchen.de/corum/ (LICENSE; CC-BY-NC)

StringDB : https://string-db.org/cgi/download.pl (LICENSE: CC-BY-4)

CellTalkDB: http://tcm.zju.edu.cn/celltalkdb/download.php (LICENSE: GNU General Public License v3.0)

### Authors

Mathieu Chevalley, GSK plc<br/>
Yusuf H Roohani, GSK plc and Stanford University<br/>
Arash Mehrjou, GSK plc<br/>
Jure Leskovec, Stanford University<br/>
Patrick Schwab, GSK plc<br/>

### Acknowledgements

MC, YR, AM and PS are employees and shareholders of GlaxoSmithKline plc.
