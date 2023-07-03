"""
Copyright (C) 2022, 2023  GlaxoSmithKline plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import importlib
import inspect
import os
import random
import sys

from causalscbench.models.abstract_model import AbstractInferenceModel


def create_experiment_folder(exp_id, output_directory):
    if exp_id == "":
        exists = True
        while exists:
            exp_id = "".join(["{}".format(random.randint(0, 9)) for _ in range(0, 6)])
            exists = os.path.exists(os.path.join(output_directory, exp_id))
    output_dir = os.path.join(output_directory, exp_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_if_valid_custom_function_file(inference_function_file_path: str):
    if inference_function_file_path == "":
        return None
    if not os.path.exists(inference_function_file_path):
        raise ValueError("The path to the custom function file does not exist.")
    else:
        module_name = "custom_acqfunc"
        spec = importlib.util.spec_from_file_location(module_name, inference_function_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        custom_inference_class = None
        for name, obj in inspect.getmembers(module, inspect.isclass):
            obj_bases = obj.__bases__
            if AbstractInferenceModel in obj_bases:
                custom_inference_class = obj
        if custom_inference_class is None:
            raise ValueError(f"No valid acquisition function was found at {inference_function_file_path}. "
                                f"Did you forget to inherit from 'AbstractInferenceModel'?")
        return custom_inference_class
