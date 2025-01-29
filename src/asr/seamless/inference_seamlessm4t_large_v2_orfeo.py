import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
import pandas as pd


def load_seamless(model_name):  #
    processor = AutoProcessor.from_pretrained(model_name)
    model = SeamlessM4Tv2Model.from_pretrained(model_name)
    return processor, model


def read_data_to_transcribe(file):
    data = pd.read_csv(file, sep='\t')
    return data["clips"]


def read_audio_and_transcribe(clip, model, processor):
    audio, orig_freq = torchaudio.load(clip + ".wav")
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq,
                                           new_freq=16_000)  # must be a 16 kHz waveform array
    audio_inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16_000)
    output = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)[0].cpu().numpy().squeeze()
    transcribe_audio = processor.decode(output, skip_special_tokens=True)
    return transcribe_audio


def main():
    processor, model = load_seamless("/home/getalp/macairec/models/pretrained_models/seamless-m4t-v2-large")
    corpus_name = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative",
                   "ofrom", "reunions", "tcof", "tufs", "valibel"]
    clips_path = "/home/getalp/macairec/data/train_dev_test_grammar_orfeo/clips_test/"
    for c in corpus_name:
        name_out = "test_" + c
        test_set = "/home/getalp/macairec/data/Orfeo_s2p/test_" + c + "_s2p.csv"
        save_file = '/home/getalp/macairec/ASR/ASR_grammar/S2P/seamless_m4t_v2/large-v2_' + name_out + "_out.txt"
        data = read_data_to_transcribe(test_set)
        f = open(save_file, 'a')
        for i in data:
            try:
                transcribe_audio = read_audio_and_transcribe(clips_path + i, model, processor)
                f.write(str(i) + '\t' + transcribe_audio + '\n')
            except:
                print("Error for this file:", str(i))
        f.close()


if __name__ == '__main__':
    main()
