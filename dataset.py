from torch.utils.data import Dataset,Subset
import pandas as pd
import numpy as np
import torch
import cv2
def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")
class KaggleADDdataset(Dataset):
    def __init__(self,data_path,transform = None):
        data_df = pd.read_parquet(data_path,engine='pyarrow')
        data_df['img_arr'] = data_df['image'].apply(dict_to_image)
        data_df.drop("image", axis=1, inplace=True)
        self.data_df = data_df
        self.transform = transform
    def __len__(self):
        return len(self.data_df)
    def __getitem__(self,item):
        image_array = self.data_df.iloc[item]['img_arr']
        label = self.data_df.iloc[item]['label']
        if label == 2:
            label = 1
        else:
            label = 0
        image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return image, label



def split_dataset_by_label(dataset):
    """
    dataset: PyTorch Dataset，返回 (x, y)
    返回:
        class0_dataset, class1_dataset
    """
    indices_class0 = []
    indices_class1 = []

    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == 0:
            indices_class0.append(i)
        elif label == 1:
            indices_class1.append(i)

    class0_dataset = Subset(dataset, indices_class0)
    class1_dataset = Subset(dataset, indices_class1)

    return class0_dataset, class1_dataset


if __name__ == "__main__":
    print('hello world!')