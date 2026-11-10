import kagglehub
import os


def download_MNIST():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Download latest version
    full_path = f"{dir_path}/MNIST"
    print(f"downloading to {full_path}")
    path = kagglehub.dataset_download("hojjatk/mnist-dataset", output_dir = full_path)
    return path

if __name__ == "__main__":
    download_MNIST()