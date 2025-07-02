import os

files_to_check = [
    '/scratch/radv/ijwamelink/classification/UCSF-PDGM-metadata_v2.xlsx',
    '/scratch/radv/ijwamelink/classification/Genetic_data.csv',
    '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/images_t1_t2_fl',
    '/data/radv/radG/RAD/users/i.wamelink/AI_benchmark/AI_benchmark_datasets/temp/609_3D-DD-Res-U-Net_Osman/testing/images_t1_t2_fl',
    '/scratch/radv/ijwamelink/classification/models/',
]

def check_path(path):
    try:
        if os.path.isdir(path):
            # Try listing contents
            os.listdir(path)
            print(f"Directory accessible: {path}")
        elif os.path.isfile(path):
            # Try opening the file
            with open(path, 'rb'):
                pass
            print(f"File accessible: {path}")
        else:
            print(f"NOT FOUND: {path}")
    except Exception as e:
        print(f"NOT ACCESSIBLE: {path} ({e})")

if __name__ == "__main__":
    for path in files_to_check:
        check_path(path)