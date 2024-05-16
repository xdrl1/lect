
import numpy
import os
import geojson
from torch.utils.data import Dataset
from utils.tif2sample import TIF2Sample
from PIL import Image
import random


# Define a triplet dataset
class TripletDataset(Dataset):
    def __init__(self, root_dir, geojson_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # handler for transformation between image coords and geo coords [lon, lat]
        # self.tif2sample = TIF2Sample(tile_w, tile_h)
        # self.tif2sample.read_tif(tif_dir)
        with open(geojson_path) as f:
            layer = geojson.load(f)
            self.geojson_list = layer["features"]

        # images and labels
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.images, self.labels = self.load_dataset()

    def load_dataset(self):
        images = []
        labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_label = self.class_to_index[class_name]

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append(img_path)
                labels.append(class_label)

        return images, labels

    # def convert_keypoint_to_lon_lat(self, img_path):
    #     str = img_path.split("\\")[-1].split(".")[0].split("_")[-2:]
    #     keypoint_i = int(str[0])
    #     keypoint_j = int(str[1])
    #     # geo_coords = self.tif2sample.transform_point(self.tif2sample.transform, keypoint_i, keypoint_j)
    #     # print(f'{geo_coords["coordinates"][0]}')

    #     return geo_coords["coordinates"][0]

    def find_coord_from_geojson(self, e_id, lst):
        index = next((idx for (idx, e) in enumerate(lst) if e["properties"]["id"] == e_id), None)
        coord = lst[index]["properties"]["left-up-coord"]["coordinates"]
        return coord[0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, anchor_idx):
        anchor_img_path = self.images[anchor_idx]
        anchor_label = self.labels[anchor_idx]

        # Select a positive sample from the same class
        positive_indices = [idx for idx, label in enumerate(self.labels) if label == anchor_label]
        positive_idx = random.choice(positive_indices)
        positive_img_path = self.images[positive_idx]
        positive_label = self.labels[positive_idx]

        # Select a negative sample from a different class
        negative_classes = [cls for cls in self.classes if cls != self.classes[anchor_label]]
        negative_class = random.choice(negative_classes)
        negative_indices = [idx for idx, label in enumerate(self.labels) if self.classes[label] == negative_class]
        negative_idx = random.choice(negative_indices)
        negative_img_path = self.images[negative_idx]
        negative_label = self.labels[negative_idx]

        # Open images
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        # Get lon, lat coords of left-up corner for each img
        # anchor_coords = self.convert_keypoint_to_lon_lat(anchor_img_path)
        # positive_coords = self.convert_keypoint_to_lon_lat(positive_img_path)
        # negative_coords = self.convert_keypoint_to_lon_lat(negative_img_path)
        anchor_id = anchor_img_path.split("/")[-1].split(".")[0]
        positive_id = positive_img_path.split("/")[-1].split(".")[0]
        negative_id = negative_img_path.split("/")[-1].split(".")[0]
        anchor_coords = self.find_coord_from_geojson(anchor_id, self.geojson_list)
        positive_coords = self.find_coord_from_geojson(positive_id, self.geojson_list)
        negative_coords = self.find_coord_from_geojson(negative_id, self.geojson_list)

        # return
        anchor = {
            "img": anchor_img,
            "id": anchor_id,
            "img_path": anchor_img_path,
            "label": anchor_label,
            "class": self.classes[anchor_label],
            "coords": anchor_coords,
        }

        positive = {
            "img": positive_img,
            "id": positive_id,
            "img_path": positive_img_path,
            "label": positive_label,
            "class": self.classes[positive_label],
            "coords": positive_coords,
        }

        negative = {
            "img": negative_img,
            "id": negative_id,
            "img_path": negative_img_path,
            "label": negative_label,
            "class": negative_class,
            "coords": negative_coords,
        }

        return anchor, positive, negative


# Define a 1v1vMany dataset
class One2ManyDataset(Dataset):
    def __init__(self, root_dir, geojson_path, nNeg=16, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.nNeg = nNeg
        with open(geojson_path) as f:
            layer = geojson.load(f)
            self.geojson_list = layer["features"]
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.images, self.labels = self.load_dataset()

    def load_dataset(self):
        images = []
        labels = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_label = self.class_to_index[class_name]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append(img_path)
                labels.append(class_label)
        return images, labels

    def find_coord_from_geojson(self, e_id, lst):
        index = next((idx for (idx, e) in enumerate(lst) if e["properties"]["id"] == e_id), None)
        coord = lst[index]["properties"]["left-up-coord"]["coordinates"]
        return coord[0]

    def LoadImg(self, imgDir):
        img = Image.open(imgDir).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, anchor_idx):
        anchor_img_path = self.images[anchor_idx]
        anchor_label = self.labels[anchor_idx]

        # Select a positive sample from the same class
        positive_indices = [idx for idx, label in enumerate(self.labels) if label == anchor_label]
        positive_idx = random.choice(positive_indices)
        positive_img_path = self.images[positive_idx]
        positive_label = self.labels[positive_idx]

        # Select nNeg samples from a different class
        imDir = numpy.array(self.images)
        label = numpy.array(self.labels)
        allNegImDir = imDir[label != anchor_label]
        allNegLabel = label[label != anchor_label]
        assert len(allNegImDir) == len(allNegLabel)
        tarIdx = random.sample(range(len(allNegImDir)), self.nNeg)
        negImDir = allNegImDir[tarIdx]
        negLabel = allNegLabel[tarIdx]

        # Open images
        ancIm = self.LoadImg(anchor_img_path)
        posIm = self.LoadImg(positive_img_path)
        negIm = []
        for imDir in negImDir:
            negIm.append(self.LoadImg(imDir))

        ancId = anchor_img_path.split("/")[-1].split(".")[0]
        posId = positive_img_path.split("/")[-1].split(".")[0]
        negId = []
        for imDir in negImDir:
            negId.append(imDir.split("/")[-1].split(".")[0])
        ancCoords = self.find_coord_from_geojson(ancId, self.geojson_list)
        posCoords = self.find_coord_from_geojson(posId, self.geojson_list)
        negCoords = []
        for thisNegId in negId:
            negCoords.append(self.find_coord_from_geojson(thisNegId, self.geojson_list))

        # return
        anchor = {
            "img": ancIm,
            "id":  ancId,
            "img_path": anchor_img_path,
            "label": anchor_label,
            "class": self.classes[anchor_label],
            "coords": ancCoords
        }

        positive = {
            "img": posIm,
            "id":  posId,
            "img_path": positive_img_path,
            "label": positive_label,
            "class": self.classes[positive_label],
            "coords": posCoords
        }

        negative = {
            "img": numpy.concatenate([numpy.array(im)[numpy.newaxis, :, :, :] for im in negIm], axis=0),
            "id": negId,
            "img_path": negImDir.tolist(),
            "label": negLabel,
            # "class": None,
            "coords": negCoords
        }

        return anchor, positive, negative


# Define a image-label dataset
class ImageLabelDataset(Dataset):
    def __init__(self, root_dir, geojson_path='', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if geojson_path != '':
            with open(geojson_path) as f:
                layer = geojson.load(f)
                self.geojson_list = layer["features"]
        # images and labels
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.label_to_class = {label: cls for cls, label in self.class_to_index.items()}
        self.images, self.labels = self.load_dataset()

    def find_coord_from_geojson(self, e_id, lst):
        index = next((idx for (idx, e) in enumerate(lst) if e["properties"]["id"] == e_id), None)
        coord = lst[index]["properties"]["left-up-coord"]["coordinates"]
        return coord[0]

    def load_dataset(self):
        images = []
        labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_label = self.class_to_index[class_name]

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append(img_path)
                labels.append(class_label)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        thisId = img_path.split("/")[-1].split(".")[0]
        coords = []
        if hasattr(self, 'geojson_list'):
            coords = self.find_coord_from_geojson(thisId, self.geojson_list)

        return image, label, coords
