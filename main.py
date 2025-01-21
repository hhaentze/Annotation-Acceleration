# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImage, SaveImage

def invert(img, keep_background=True):
    img = img.clip(0, 3000)
    _max = img.max()
    _min = img.min()
    img = _max - img + _min

    if keep_background:
        th = (_max - _min) * 0.99 + _min
        img[img >= th] = 0

    return img

def main():
    # Check Input Arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        sys.exit(1)

    allowed_extensions = {".nii", ".nii.gz", ".mha"}
    if not any(image_path.endswith(ext) for ext in allowed_extensions):
        print(f"Error: Unsupported file extension. Allowed extensions are: {allowed_extensions}")
        sys.exit(1)

    # Load Image
    load_transform = LoadImage(image_only=True)
    img = load_transform(image_path)

    # Invert Image
    inverted_img = invert(img)

    # Save Inverted Image
    save_transform = SaveImage(output_dir=".", output_postfix="inverted", output_ext=".nii.gz")
    save_transform(inverted_img, image_path)

if __name__ == "__main__":
    main()
