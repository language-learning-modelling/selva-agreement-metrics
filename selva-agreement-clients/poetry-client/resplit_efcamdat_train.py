import pathlib
import json
import os
def write_batch(batch,outfp):
    with open(outfp,"w") as outf:
        json.dump(batch, outf)


splits_train_folder = pathlib.Path("./outputs/EFCAMDAT/splits/train_json/original_split/")

MAX_BATCH_SIZE=5000
OUTPUT_FOLDER="./outputs/EFCAMDAT/splits/train_json/subsplits/"
for fp in splits_train_folder.iterdir():
    batch_id = None
    filename=os.path.basename(fp)
    batch = {}
    data = json.load(open(fp))
    for (idx, (instance_id, instance)) in enumerate(data.items()):
        batch[idx] = instance
        print(len(batch))
        if len(batch) >= MAX_BATCH_SIZE:
            print(idx+1)
            batch_id = str((idx+1)// MAX_BATCH_SIZE).zfill(3)
            output_filename=filename.replace(".json",f"_{batch_id}.json")
            outfp=f"{OUTPUT_FOLDER}/{output_filename}"
            write_batch(batch, outfp)
            batch = {}
    if batch:
        batch_id = str(int(batch_id)+1).zfill(3) if batch_id else str(1).zfill(3)
        output_filename=filename.replace(".json",f"_{batch_id}.json")
        outfp=f"{OUTPUT_FOLDER}/{output_filename}"
        write_batch(batch, outfp)
