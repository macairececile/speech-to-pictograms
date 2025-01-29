#!/usr/bin/python

import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import whisper


def read_tsv(tsv_file):
    return pd.read_csv(tsv_file, sep='\t')


def run_whisper_and_get_prediction(test_set, clips_path, save_file, index):
    data = read_tsv(test_set)
    clips = data['clips'].values.tolist()[index:]
    model = whisper.load_model("/home/getalp/macairec/large-v3.pt")
    f = open(save_file, 'a')
    for i, c in enumerate(clips):
        try:
            result = model.transcribe(clips_path + c + '.wav', language='fr')
            f.write(c + '\t' + result["text"] + '\n')
        except:
            print("Error for this file:", str(c))
    f.close()


def asr_whisper_orfeo():
    corpus_name = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative",
                   "ofrom", "reunions", "tcof", "tufs", "valibel"]
    for c in corpus_name:
        name_out = "test_" + c
        test_set = "/home/getalp/macairec/data/Orfeo_s2p/test_" + c + "_s2p.csv"
        clips_path = "/home/getalp/macairec/data/train_dev_test_grammar_orfeo/clips_test/"
        save_file = '/home/getalp/macairec/ASR/ASR_grammar/S2P/large-v3_' + name_out + "_out.txt"
        run_whisper_and_get_prediction(test_set, clips_path, save_file, 0)


if __name__ == '__main__':
    asr_whisper_orfeo()
