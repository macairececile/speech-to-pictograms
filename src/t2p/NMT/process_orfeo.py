import pandas as pd

SPLITS = ["train", "valid", "test"]


def read_data_from_grammar(csv_file):
    return pd.read_csv(csv_file, sep="\t")


def get_src_and_tgt_from_data(data):
    return data["text"], data["tgt"]


def create_file(split, sentences, src_or_tgt):
    f = open(split + "." + src_or_tgt, "w")
    for s in sentences:
        f.write(str(s) + "\n")
    f.close()


def get_text_and_generate_file(folder):
    src = "fr"
    tgt = "frp"
    train_valid_src = []
    train_valid_tgt = []
    for i in SPLITS:
        data = read_data_from_grammar(folder + i + "_s2p_orfeo.csv")
        src_sentences, tgt_sentences = get_src_and_tgt_from_data(data)
        if i in ["train", "valid"]:
            train_valid_src.extend(src_sentences)
            train_valid_tgt.extend(tgt_sentences)
        create_file(i, src_sentences, src)
        create_file(i, tgt_sentences, tgt)
    create_file("train_valid", train_valid_src, src)
    create_file("train_valid", train_valid_tgt, tgt)


if __name__ == '__main__':
    get_text_and_generate_file("/gpfswork/rech/czj/uef37or/data/orfeo/")
