import subprocess
import os
import sys

def extract_audio(video_file,output_ext="wav"):
    base=os.path.basename(video_file)
    name,ext=os.path.splitext(video_file)

    audio_name=name+".wav"
    output_path=os.path.join("data/processed/audio",audio_name)
    subprocess.call(
        ["ffmpeg",
         "-y",
         "-i",
         "video_file",
         f"{name}.{output_ext}"],
         stdout=subprocess.DEVNULL,
         stderr=subprocess.STDOUT
    )

if __name__ == "__main__":
    vf = sys.argv[1]
    extract_audio(vf)