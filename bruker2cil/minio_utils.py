# -*- coding: utf-8 -*-
#  Copyright 2024 Till Leissner
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#   Authored by: Till Leissner (University of Southern Denmark)
#   Edited by: 
#
#   We acknowledge support from the ESS Lighthouse on Hard Materials in 3D, SOLID, funded by the Danish Agency for Science and Higher Education (grant No. 8144-00002B).


### General imports 
from minio import Minio
from minio.error import S3Error
from PIL import Image
import io

def list_tiff_images(bucket: str, prefix: str = ""):
    """List TIFF image filenames under a specific prefix (folder) in MinIO."""
    return [
        obj.object_name
        for obj in client.list_objects(bucket, prefix=prefix, recursive=False)
        if obj.object_name.lower().endswith((".tif", ".tiff"))
    ]

def load_image(bucket: str, object_name: str):
    """
    Load an image file from MinIO into a PIL Image.
    """
    response = client.get_object(bucket, object_name)
    image = Image.open(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    return image

def load_all_tiff_images(bucket: str, prefix: str = ""):
    """
    Load all TIFF images from MinIO bucket under a specific prefix.
    Returns list of (filename, PIL.Image) tuples.
    """
    file_list = list_tiff_images(bucket, prefix)
    return [(name, load_image(bucket, name)) for name in file_list]

