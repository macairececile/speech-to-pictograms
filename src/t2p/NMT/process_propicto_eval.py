import pandas as pd

def read_data_from_grammar(csv_file):
    return pd.read_csv(csv_file, sep="\t")


def get_src_and_tgt_from_data(data):
    return data["text"], data["tgt"]


def create_file(split, sentences, src_or_tgt):
    f = open(split + "." + src_or_tgt, "w")
    for s in sentences:
        f.write(str(s) + "\n")
    f.close()


def get_text_and_generate_file():
    src = "fr"
    tgt = "frp"
    data = read_data_from_grammar("/gpfswork/rech/czj/uef37or/data/propicto-eval/propicto_eval.csv")
    src_sentences, tgt_sentences = get_src_and_tgt_from_data(data)
    create_file("propicto_eval", src_sentences, src)
    create_file("propicto_eval", tgt_sentences, tgt)


if __name__ == '__main__':
    get_text_and_generate_file()
