import os
def unset_env_var(var):
    if var in os.environ:
        del os.environ[var]


unset_env_var('http_proxy')
unset_env_var('https_proxy')
unset_env_var('no_proxy')

path = "/Users/yingdong/Downloads/"

from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf # https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/pdf.py#L119

# Get elements
raw_pdf_elements = partition_pdf(
    filename="/Users/yingdong/Downloads/Approved_Cryptographies.pdf",
    # Unstructured first finds embedded image blocks
    extract_images_in_pdf=False,
    # Use a layout detection model to identify document elements
    strategy="hi_res",
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    max_characters=1200, # 4000, # 600,
    new_after_n_chars=1000, # 3800, # 450,
    combine_text_under_n_chars=500, # 2000, # 150,
    image_output_dir_path=path,
    mode="elements"
)
