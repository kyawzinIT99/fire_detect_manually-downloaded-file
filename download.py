from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

output_path = "./kernel_output"
os.makedirs(output_path, exist_ok=True)

api.kernels_output("unalkucukcan/firedetection-with-cnn-proje-unalkucukcan", path=output_path)
print("✅ Kernel output downloaded")
