from speechbrain.inference.ASR import EncoderASR
import pandas as pd
# import evaluate

# wer_metric = evaluate.load("wer")


def extract_data(tsv_file):
    data = pd.read_csv(tsv_file, sep='\t')
    return data["clips"].tolist(), data["text"].tolist()


def generate_transcriptions_from_inference_model(clips, model):
    transcripts = []
    for i in clips:
        transcripts.append(model.transcribe_file("/gpfswork/rech/czj/uef37or/data/propicto-eval/clips/" + i + ".wav").lower())
    return transcripts


def save_transcripts(transcripts, clips, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, t in enumerate(transcripts):
            file.write(f"{clips[i]}\t{t}\n")


def compute_wer(transcripts, refs):
    print("WER score : ", wer_metric.compute(predictions=transcripts, references=refs))


def main():
    model = EncoderASR.from_hparams(
        "/gpfswork/rech/czj/uef37or/ASR_speechbrain_wav2vec2/wav2vec2_7k_large/save/CKPT+2023-12-20+15-33-48+00/",
        savedir="tmp/"
    )
    clips, refs = extract_data("/gpfswork/rech/czj/uef37or/data/propicto-eval/propicto_eval.csv")
    trans = generate_transcriptions_from_inference_model(clips, model)
    save_transcripts(trans, clips,
                     "/gpfswork/rech/czj/uef37or/ASR_S2P_data/propicto_eval/wav2vec2_orfeo_propicto_eval_out.txt")
    # compute_wer(trans, refs)


if __name__ == '__main__':
    main()
