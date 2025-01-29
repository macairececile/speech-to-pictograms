import pandas as pd

corpus_name = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions",
               "tcof", "tufs", "valibel"]


def read_data_from_grammar(csv_file):
    return pd.read_csv(csv_file, sep="\t")


def get_text_and_generate_file(folder, outdir):
    for c in corpus_name:
        data = read_data_from_grammar(folder + "test_" + c + "_s2p.csv")
        ids = []
        translation = []
        for n, row in data.iterrows():
            ids.append(row["clips"])
            translation.append({"fr": row["text"], "frp": row["tgt"]})
        file = pd.DataFrame({"id": ids, "translation": translation})
        file.to_csv(outdir + "test_" + c + ".csv", sep='\t', index=False)


if __name__ == '__main__':
    get_text_and_generate_file("/gpfswork/rech/czj/uef37or/data/orfeo/",
                               "/gpfswork/rech/czj/uef37or/Fine_tuning_t5/data_orfeo/")
