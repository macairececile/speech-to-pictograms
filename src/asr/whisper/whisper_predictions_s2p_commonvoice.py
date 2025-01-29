#!/usr/bin/python

import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import whisper


def read_tsv(tsv_file):
    return pd.read_csv(tsv_file, sep='\t')


def run_whisper_and_get_prediction(test_set, clips_path, save_path, index):
    data = read_tsv(test_set)
    clips = data['clips'].values.tolist()[index:]
    model = whisper.load_model("/home/getalp/macairec/large-v3.pt")
    name_out = test_set.split("/")[-1].split('.csv')[0]
    f = open('/home/getalp/macairec/ASR/ASR_grammar/S2P/large-v3_' + name_out + "_out.txt", 'a')
    for i, c in enumerate(clips):
        try:
            result = model.transcribe(clips_path + c + '.mp3', language='fr')
            f.write(c + '\t' + result["text"] + '\n')
        except:
            print("Error for this file:", str(c))
    f.close()


def asr_whisper_cfpp(args):
    run_whisper_and_get_prediction(args.data, args.clips, args.output, int(args.index))


if __name__ == '__main__':
    parser = ArgumentParser(description="Run whisper and get predictions and wer + cer scores",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help="")
    parser.add_argument('--clips', type=str, required=True,
                        help="")
    parser.add_argument('--output', type=str, required=True,
                        help="")
    parser.add_argument('--index', type=str, required=True,
                        help="")
    parser.set_defaults(func=asr_whisper_cfpp)
    args = parser.parse_args()
    args.func(args)
