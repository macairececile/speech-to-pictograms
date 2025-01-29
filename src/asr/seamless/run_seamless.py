import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
from argparse import ArgumentParser, RawTextHelpFormatter

punctuations = '!?\":,/.;()[]'
punc_table = str.maketrans({key: None for key in punctuations})

# usage : python run_seamless.py --model "seamless-m4t-v2-large" --audio "2670105_230307_194609622.wav"

def process_pred(pred):
    """
        Nettoyer la prédiction du modèle.

        Arguments
        ---------
        pred : str
    """
    ref = pred.replace('- ', ' ').replace(' -', ' ')
    ref = str(ref).translate(punc_table)  # enlever la ponctuation
    process = ref.lower()  # mettre en minuscule
    return process.strip()  # enlever les espaces inutiles


def load_seamless(model_name):
    """
        Appelle le modèle seamless.

        Arguments
        ---------
        model_name : str
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = SeamlessM4Tv2Model.from_pretrained(model_name)
    return processor, model


def run_seamless(args):
    """
        Applique le modèle et récupère la prédiction.
    """
    processor, model = load_seamless(args.model)
    audio, orig_freq = torchaudio.load(args.audio)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq,
                                           new_freq=16_000)  # must be a 16 kHz waveform array
    audio_inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16_000)
    output = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)[
        0].cpu().numpy().squeeze()  # génère la pred
    transcribe_audio = processor.decode(output, skip_special_tokens=True)
    out = process_pred(transcribe_audio)
    return out


if __name__ == '__main__':
    parser = ArgumentParser(description="Run seamless and get the prediction",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help="")
    parser.add_argument('--audio', type=str, required=True,
                        help="")
    parser.set_defaults(func=run_seamless)
    args = parser.parse_args()
    args.func(args)
