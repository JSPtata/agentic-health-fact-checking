import subprocess
import os
import sys
import json
import whisper #type:ignore
import cv2 #type:ignore
import pytesseract #type:ignore
import difflib

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
    
    segments=[]

    for segment in result["segments"]:
        segments.append(segment["text"])
        segments.append(segment["start"])
        segments.append(segment["end"])

    print(segments)
    return result

def ocr_extraction(video_file):
    os.makedirs("data/processed/frames", exist_ok=True)
    vidcap=cv2.VideoCapture(video_file)
    fps=vidcap.get(cv2.CAP_PROP_FPS)
    interval=int(fps*2)

    count=0
    success,image=vidcap.read()

    ocr_results=[]


    while success:
        if count%interval==0:
            frame_path=os.path.join("data/processed/frames",f"frame{count}.jpg")
            cv2.imwrite(frame_path,image)
            
            text = pytesseract.image_to_string(image).strip()
            time_stamp=count/fps

            
            if text and len(text)>20:
                if(len(ocr_results)==0):
                    ocr_results.append(
                        {
                            "text":text,
                            "timestamp":time_stamp
                        }
                    )
                else:
                    similarity=difflib.SequenceMatcher(None,text,ocr_results[-1]["text"]).ratio()
                    if(similarity<0.90):
                        ocr_results.append(
                            {
                                "text":text,
                                "timestamp":time_stamp
                            }
                        )
        count+=1
        success,image=vidcap.read()
    
    vidcap.release()
    return ocr_results
        

def save_transcript(result,audio_file,ocr_results):
    video_file=os.path.basename(audio_file)
    video_id,ext=os.path.splitext(video_file)

    segments=[]

    for segment in result["segments"]:
        one_segment={
            "text":segment["text"],
            "start":segment["start"],
            "end":segment["end"]
        }
        segments.append(one_segment)

    d={
        "video_id":video_id,
        "language":result["language"],
        "text":result["text"],
        "segments":segments,
        "on_screen_text":ocr_results
    }

    output_file=video_id+".json"
    output_path=os.path.join("data/processed/transcripts",output_file)

    with open(output_path,"w") as file:
        json.dump(d,file,indent=4)


    print(d)





if __name__ == "__main__":
    vf = sys.argv[1]
    audio_path=extract_audio(vf)
    result=transcribe_audio(audio_path)
    ocr_results=ocr_extraction(vf)
    save_transcript(result,audio_path,ocr_results)
    for item in ocr_results:
        print("\nTimestamp:", item["timestamp"])
        print(item["text"])