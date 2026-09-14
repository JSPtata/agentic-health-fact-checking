import subprocess
import os
import sys
import whisper #type:ignore

def extract_audio(video_file,output_ext="wav"):
    base=os.path.basename(video_file)
    name,ext=os.path.splitext(base)

    audio_name=name+".wav"
    output_path=os.path.join("data/processed/audio",audio_name)
    subprocess.call(
        ["ffmpeg",
         "-y",
         "-i",
         video_file,
         f"{output_path}"],
         stdout=subprocess.DEVNULL,
         stderr=subprocess.STDOUT
    )

    return output_path


def transcribe_audio(audio_file):
    model=whisper.load_model("small")
    result=model.transcribe(audio_file)
    print(result["text"])
    print(result.keys())
    print(result["segments"])
    return result


if __name__ == "__main__":
    vf = sys.argv[1]
    audio_path=extract_audio(vf)
    result=transcribe_audio(audio_path)