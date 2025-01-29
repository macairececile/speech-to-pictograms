import pandas as pd
import ast


def read_tsv_file(tsv_file):
    """
    Function to load a tsv file
    :param tsv_file: Name of the tsv file to load
    :type tsv_file: str
    :returns: dataframe object
    """
    return pd.read_csv(tsv_file, sep='\t')


def select_data_from_dataframe_and_add_ref_sense_keys_to_dict(dataframe, out_asr_for_wsd):
    """
        Function to add sense_keys for each disambiguate sentence
        :param dataframe: dataframe with the sense keys
        :param out_asr_for_wsd: dict with asr result before wsd
        :returns: dict with asr output and sense keys associated to asr output
        """
    data = dataframe[dataframe.path.isin(out_asr_for_wsd.keys())]
    for k, v in out_asr_for_wsd.items():
        index = 0
        for i, el in enumerate(v[0]):
            if el[0] != '<eps>':
                if not ast.literal_eval(data.loc[data['path'] == k]["sense_keys"].values[0])[i + index]:
                    el.append([])
                else:
                    el.append(ast.literal_eval(data.loc[data['path'] == k]["sense_keys"].values[0])[i + index])
            else:
                el.append([])
                index -= 1
    return out_asr_for_wsd


def read_txt_file(txt_file):
    """
        Function to read txt lines from ASR output from speechbrain model
        :param txt_file: text file with asr output
        :returns: str lines from text file
    """
    with open(txt_file) as f:
        lines = f.readlines()
    return lines


def get_line(line, index, inc):
    """
        Function to get line per index per increment and remove eps + line break
        :param line: line of the text file
        :param index : index of the line
        :param inc : increment
        :returns: str line
    """
    l = ''.join(''.join(line[index + inc].split(';')).split('<eps>')).replace('\n', '')
    l = ' '.join(l.split())
    return l


def linguistic_processing(sentence):
    """
        Function to clean the sentence (replace special characters)
        :param sentence: string
        :returns: cleaned sentence (str)
    """
    if type(sentence) == str:
        sentence = sentence.lower()
        particules = ['qu ', 'c ', 'd ', 'n ', ' hui', 'j ', 'l ', 's ', 't ']
        replacements = {' qu ': " qu'", ' c ': " c'", ' d ': " d'", ' n ': " n'", ' hui ': "'hui ", ' j ': " j'",
                        ' l ': " l'", ' s ': " s'", ' t ': " t'"}
        for el in particules:
            if sentence.startswith(el):
                sentence = sentence[:len(el)].replace(el, el[:-1] + "'") + sentence[len(el):]
        for k, v in replacements.items():
            sentence = sentence.replace(k, v)
    return sentence


def process_out_asr_and_get_info(file):
    """
        Function to process the output file from asr and retrieve specific info (wer, hyp, ref)
        :param file: text file with the asr output
        :returns: dictionary with the specified information by id (name of the utterance file)
    """
    lines = read_txt_file(file)
    out_asr_process = {}
    for x in range(12, len(lines), 5):
        utt_id = lines[x].split(',')[0]
        wer = lines[x].split('WER ')[1][:6].split('[')[0]
        ref = linguistic_processing(get_line(lines, x, 1))
        hyp = linguistic_processing(get_line(lines, x, 3))
        out_asr_process[utt_id] = [float(wer), ref, hyp]
    return out_asr_process


def process_out_asr_to_evaluate_wsd(file, corpus_eval):
    """
        Function to process the output file from asr and retrieve specific info (wer, hyp, ref) to evaluate the disambiguation from asr output
        :param file: text file with the asr output
        :param corpus_eval : tsv file with the annotated corpus with sense keys per utterance
        :returns: dictionary with the specified information by id (name of the utterance file)
    """
    data_ref = read_tsv_file(corpus_eval)
    lines = read_txt_file(file)
    out_wsd_pictos_with_refs = {}
    for x in range(12, len(lines), 5):
        utt_id = lines[x].split(',')[0]
        words_ref = [r.strip() for r in lines[x + 1].replace('\n', '').split(" ; ")]
        words_hyp = [r.strip() for r in lines[x + 3].replace('\n', '').split(" ; ")]
        hyp = linguistic_processing(get_line(lines, x, 3))
        out_wsd_pictos_with_refs[utt_id] = [[list(a) for a in zip(words_ref, words_hyp)], hyp]

    return select_data_from_dataframe_and_add_ref_sense_keys_to_dict(data_ref, out_wsd_pictos_with_refs)
