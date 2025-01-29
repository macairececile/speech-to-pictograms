import pandas as pd

SPLITS = ["train", "valid", "test"]


def read_data_from_grammar(csv_file):
    return pd.read_csv(csv_file, sep="\t")


def get_text_and_generate_file(folder, outdir):
    for i in SPLITS:
        data = read_data_from_grammar(folder + i + "_tedx.csv")
        # data = read_data_from_grammar(folder + "polylexical.csv")
        ids = []
        translation = []
        for n, row in data.iterrows():
            ids.append(row["clips"])
            translation.append({"fr": row["text"], "frp": row["tokens"]})
        file = pd.DataFrame({"id": ids, "translation": translation})
        file.to_csv(outdir + i + "_tedx.csv", sep='\t', index=False)
        # file.to_csv(outdir + "polylexical.csv", sep='\t', index=False)


if __name__ == '__main__':
    get_text_and_generate_file("/gpfswork/rech/czj/uef37or/data/corpus_v2/tedx/",
                               "/gpfswork/rech/czj/uef37or/data/corpus_v2/data_hf_finetuning/")
