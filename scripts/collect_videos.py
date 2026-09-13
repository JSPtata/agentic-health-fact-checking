import csv 
import yt_dlp #type:ignore

ydl_opts={}

def csv_to_list(filename):
    result_list=[]
    with open(filename) as file_obj:
        reader=csv.DictReader(file_obj,delimiter=",")
        for row in reader:
            result_list.append(row)
    
    return result_list

rows=csv_to_list("data/manifest_v2.csv")


for row in rows:
    if(row["status"]=="pending"):
        url=row["url"]
        video_id=row["video_id"]

        ydl_opts={
            "outtmpl":f"data/raw/videos/{video_id}.%(ext)s"
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url,download=True)
            row["status"]="downloaded"
        except yt_dlp.utils.DownloadError as error:
            row["status"]="download_error"
            row["notes"]=str(error)


        
with open("data/manifest_v2.csv","w",newline="") as file:
    writer=csv.DictWriter(file,fieldnames=["video_id","category","url","status","notes"])
    writer.writeheader() 
    writer.writerows(rows) 