import shutil
from pathlib import Path

## script to autmatically updates __init__.py when adding new datasets.


### copy new datasets to mmsegmentation folder
new_dataset_folder = Path("./new_datasets")
dst_dataset_folder = Path("./Pointcept") / "pointcept" / "datasets"
for new_dataset in new_dataset_folder.glob("*.py"):
    shutil.copy(src = str(new_dataset), dst=str(dst_dataset_folder / new_dataset.name))


### update init.py
## dinstinguish imports and __all__
new_lines = []
other_lines = []
all_bool = False
with open(str(dst_dataset_folder / "__init__.py"),"r") as f:
    for line in f:
        if line.startswith("__all__") or all_bool:
            all_bool = True
            other_lines.append(line)
        else:
            new_lines.append(line)

### update new lines with new datasets
for new_dataset in new_dataset_folder.glob("*.py"):
    class_name = ""
    with open(str(new_dataset),"r") as f:
        for line in f:
            if line.startswith("class "):
                class_name = line.split('(')[0].replace("class ","")

    new_line = "from .%s import %s\n"%(new_dataset.stem, class_name)
    ## only add if it it does not exist yet
    if not new_line in new_lines:
        new_lines.append(new_line)

## write file
with open(str(dst_dataset_folder / "__init__.py"),"w") as f:
    for line in new_lines:
        f.write(line)
    for line in other_lines:
        f.write(line)
print("Succesfully updated __init__py")
# for new_dataset in new_dataset_folder.glob("*.py"):
#     new
