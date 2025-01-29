#!/usr/bin/env python3
# Copyright (c) ByteDance, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os.path as op
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import torchaudio
from data_utils import filter_manifest_df, gen_config_yaml, gen_vocab, save_df_to_tsv

from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "speaker",
                    "src_text", "tgt_text", "src_lang", "tgt_lang"]

SPLITS = ["propicto_eval"] # ["test_s2p_orfeo"] # "train_s2p_orfeo", "valid_s2p_orfeo",


def read_data_from_grammar(csv_file):
    return pd.read_csv(csv_file, sep="\t")


def get_duration_file(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    duration = waveform.size(1) / sample_rate
    return duration


def generate_yaml(out_grammar_file, type, wav_path, data_path):
    f = open(data_path + type + ".yaml", "w")
    for i, row in out_grammar_file.iterrows():
        audio_file_path = wav_path + row["clips"] + ".wav"
        duration = get_duration_file(audio_file_path)
        line = "- {duration: " + str(duration) + ", offset: 0.0, speaker_id: spk.0, wav: " + audio_file_path + "}"
        f.write(line + "\n")
    f.close()


def process_orfeo(split, wav_root, grammar_file, data_path):
    data = []
    import yaml
    with open(data_path + split + ".yaml") as f:
        segments = yaml.load(f, Loader=yaml.BaseLoader)

    for _lang in ["fr", "frp"]:
        if _lang == "fr":
            utterances = grammar_file["text"]
        else:
            utterances = grammar_file["tgt"]
        for i, u in enumerate(utterances):
            segments[i][_lang] = u
    for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
        wav_path = op.join(wav_root, wav_filename)
        sample_rate = torchaudio.info(wav_path).sample_rate
        seg_group = sorted(_seg_group, key=lambda x: x["offset"])
        for i, segment in enumerate(seg_group):
            if split in ["train_s2p_orfeo", "valid_s2p_orfeo"]:
                if float(segment["duration"]) >= 2.0:
                    offset = int(float(segment["offset"]) * sample_rate)
                    n_frames = int(float(segment["duration"]) * sample_rate)
                    _id = wav_filename.split(".wav")[0]
                    data.append((wav_path, offset, n_frames, sample_rate, segment["fr"], segment["frp"],
                                 segment["speaker_id"], _id,))
            else:
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = wav_filename.split(".wav")[0]
                data.append((wav_path, offset, n_frames, sample_rate, segment["fr"], segment["frp"],
                             segment["speaker_id"], _id,))
    return data


def process(data_root, wav_root, vocab_type, vocab_size, outputdir):
    train_text = []
    for split in SPLITS:
        grammar_file = read_data_from_grammar(data_root + split + ".csv")
        generate_yaml(grammar_file, split, wav_root, outputdir)
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = process_orfeo(split, wav_root, grammar_file, outputdir)
        for wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id in tqdm(dataset):        
            manifest["id"].append(utt_id)
            manifest["audio"].append(f"{wav_path}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(spk_id)
            manifest["src_lang"].append("fr")
            manifest["tgt_lang"].append("frp")
            manifest["src_text"].append(src_utt)
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
            train_text.extend(manifest["src_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=1000, max_n_frames=480000)
        save_df_to_tsv(df, op.join(outputdir, f"{split}_st.tsv"))

    # v_size_str = "" if vocab_type == "char" else str(vocab_size)
    # spm_filename_prefix = f"spm_{vocab_type}{v_size_str}_st"
    # with NamedTemporaryFile(mode="w") as f:
    #     for t in train_text:
    #         f.write(str(t) + "\n")
    #     gen_vocab(
    #         f.name,
    #         op.join(outputdir, spm_filename_prefix),
    #         vocab_type,
    #         vocab_size,
    #         accept_language=["fr", "frp"],
    #         user_defined_symbols=["<lang:fr>", "<lang:frp>"],
    #     )
    # # Generate config YAML
    # gen_config_yaml(
    #     outputdir,
    #     spm_filename_prefix + ".model",
    #     yaml_filename=f"config_st.yaml",
    #     prepend_tgt_lang_tag=True,
    #     prepend_src_lang_tag=True
    # )


if __name__ == '__main__':
    # process("/gpfswork/rech/czj/uef37or/data/orfeo/",
    #         "/gpfsscratch/rech/czj/uef37or/clips_orfeo/", "unigram", 10000,
    #         "/gpfswork/rech/czj/uef37or/ConST/orfeo_process/")
    process("/gpfswork/rech/czj/uef37or/data/propicto-eval/",
            "/gpfswork/rech/czj/uef37or/data/propicto-eval/clips/", "unigram", 10000,
            "/gpfswork/rech/czj/uef37or/ConST/")
