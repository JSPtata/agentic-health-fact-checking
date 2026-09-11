import csv 
import yt_dlp #type:ignore

ydl_opts={}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    with open("data/manifest_v2.csv") as file:
        reader=csv.DictReader(file)

        for row in reader:
            if row["status"]=="pending":
                video_id=row["video_id"]
                url=row["url"]

                output=f"data/raw/videos/{video_id}.%(ext)s"

                ydl_opts={
                    'outtmpl':output
                }
                print("Downloading:",row["video_id"])
                ydl.download([url])
                break

