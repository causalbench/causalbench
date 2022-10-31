"""
Copyright 2021 GSK plc

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
import os
import random


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
