""" Module for converting trained models (i.e. optimizing models) from caffe framework using Intel OpenVino

This module convert models  (i.e. perform model optimization) from caffe framework
and output an Intermediate Representation (IR) of the model. 
The converted model can be test using the Intel OpenVino Inference Engine and/or deployed to the 
desired target (Intel Movidius, Luxonis device...)

Attributes:
    BASE_DIR (Path): base directory
    model_path (Path): the path to the .caffemodel file
    model_proto_path (Path): the path to the .prototxt file
    model_mean_path (Path): the path to the .binaryproto mean file

Usage:
    Compile all the required attributes and the run. The converted model will be saved (if not already existing)
    in the same folder where the original model is. Three files will be created:
        * *.bin
        * *.mapping
        * *.xml
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# set here the models you wish to convert
# mode
model_path = BASE_DIR.joinpath("model.caffemodel")
model_proto_path = BASE_DIR.joinpath("model.prototxt")
model_mean_path = BASE_DIR.joinpath("mean.binaryproto")

ir_path = Path(model_path).with_suffix(".xml")
print("Selected model {model_path}")
mo_command = f"""mo
                 --input_model "{model_path}" 
                 --input_proto "{model_proto_path}"
                 --mean_file "{model_mean_path}"
                 --scale_values="[127.5]" 
                 --data_type FP16 
                 --output_dir "{model_path.parent}"
                 """
mo_command = " ".join(mo_command.split())
if not ir_path.exists():
    print("Exporting model to IR... This may take a few minutes.")
    os.system(mo_command)
else:
    print(f"IR model {ir_path} already exists.")
