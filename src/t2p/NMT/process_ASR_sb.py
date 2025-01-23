import csv

def get_data_from_asr(path_file_asr, corpus_name):
    """
    Read the data generated by whisper.

    Args:
        directory (str): The directory path.
        model_prefix (str): The prefix for the model.
        corpus_name (str): The name of the corpus.

    Returns:
        list: The predictions of the whisper model.
    """
    file_path = path_file_asr + "test_"+corpus_name+"_out.txt"
    f = open(file_path, 'r')
    preds = [row[1] for row in csv.reader(f, delimiter='\t')]
    f.close()
    return preds


def create_file(sentences, corpus_name, folder_to_save):
    f = open(folder_to_save + "test_wav2vec2_"+ corpus_name + ".fr", "w")
    for s in sentences:
        f.write(str(s) + "\n")
    f.close()


def get_text_and_generate_file(path_file_asr, corpus_name, folder_to_save):
    src_sentences = get_data_from_asr(path_file_asr, corpus_name)
    create_file(src_sentences, corpus_name, folder_to_save)


if __name__ == '__main__':
    get_text_and_generate_file("/gpfswork/rech/czj/uef37or/ASR_S2P_data/wav2vec2_ctc_greedy/","commonvoice", "/gpfswork/rech/czj/uef37or/NMT_transformers/exp_commonvoice/commonvoice_process/")
    # get_text_and_generate_file("/gpfswork/rech/czj/uef37or/ASR_S2P_data/wav2vec2_ctc_greedy/", "orfeo", "/gpfswork/rech/czj/uef37or/NMT_transformers/exp_orfeo/orfeo_process/")
