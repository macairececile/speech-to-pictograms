import pandas as pd

corpus_name = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof", "tufs", "valibel"]

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
    for i in corpus_name:
        data = read_data_from_grammar(folder + "test_" + i + "_s2p.csv")
        src_sentences, tgt_sentences = get_src_and_tgt_from_data(data)
        create_file(i, src_sentences, src)
        create_file(i, tgt_sentences, tgt)


if __name__ == '__main__':
    get_text_and_generate_file("/gpfswork/rech/czj/uef37or/data/orfeo/")
