import csv
import json
import os
import datasets

_CITATION = """\
@inproceedings{juraska-etal-2019-viggo,
    title = "{V}i{GGO}: A Video Game Corpus for Data-To-Text Generation in Open-Domain Conversation",
    author = "Juraska, Juraj  and
      Bowden, Kevin  and
      Walker, Marilyn",
    booktitle = "Proceedings of the 12th International Conference on Natural Language Generation",
    month = oct # "{--}" # nov,
    year = "2019",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-8623",
    doi = "10.18653/v1/W19-8623",
    pages = "164--172",
}
"""

_DESCRIPTION = """\
ViGGO was designed for the task of data-to-text generation in chatbots (as opposed to task-oriented dialogue systems), with target responses being more conversational than information-seeking, yet constrained to the information presented in a meaning representation. The dataset, being relatively small and clean, can also serve for demonstrating transfer learning capabilities of neural models.
"""

_URLs = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
    "challenge_train_1_percent": "challenge_train_1_percent.csv",
    "challenge_train_2_percent": "challenge_train_2_percent.csv",
    "challenge_train_5_percent": "challenge_train_5_percent.csv",
    "challenge_train_10_percent": "challenge_train_10_percent.csv",
    "challenge_train_20_percent": "challenge_train_20_percent.csv",
}


class Viggo(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "viggo"

    def _info(self):
        features = datasets.Features(
            {
                "gem_id": datasets.Value("string"),
                "meaning_representation": datasets.Value("string"),
                "target": datasets.Value("string"),
                "references": [datasets.Value("string")],
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=datasets.info.SupervisedKeysData(
                input="meaning_representation", output="target"
            ),
            homepage="https://nlds.soe.ucsc.edu/viggo",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=spl, gen_kwargs={"filepath": dl_dir[spl], "split": spl}
            )
            for spl in _URLs.keys()
        ]

    def _generate_examples(self, filepath, split, filepaths=None, lang=None):
        """Yields examples."""
        with open(filepath, "r", encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for id_, row in enumerate(reader):
                yield id_, {
                    "gem_id": f"viggo-{split}-{id_}",
                    "meaning_representation": row["mr"],
                    "target": row["ref"],
                    "references": [row["ref"]],
                }
