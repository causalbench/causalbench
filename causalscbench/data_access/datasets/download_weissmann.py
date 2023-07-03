"""
Copyright (C) 2022  GlaxoSmithKline plc - Mathieu Chevalley;

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
from causalscbench.data_access.utils import download


def download_weissmann_k562(output_directory):
    URL = "https://plus.figshare.com/ndownloader/files/35773219"
    filename = "k562.h5ad"
    path = download.download_if_not_exist(URL, output_directory, filename)
    return path

def download_weissmann_rpe1(output_directory):
    URL = "https://plus.figshare.com/ndownloader/files/35775606"
    filename = "rpe1.h5ad"
    path = download.download_if_not_exist(URL, output_directory, filename)
    return path

def download_summary_stats(output_directory):
    URL = "https://ars.els-cdn.com/content/image/1-s2.0-S0092867422005979-mmc2.xlsx"
    filename = "summary_stats.xlsx"
    path = download.download_if_not_exist(URL, output_directory, filename)
    return path
