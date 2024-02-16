import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np 
import re
import json
from csv import DictReader

from transformers import AutoTokenizer
from torchvision import transforms
from pyvi import ViTokenizer

# for image: (224, 224) -> (224, 224, 3)

# label setting:
# 0 for NORMAL, 1 for PNEUMONIA

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

class ViVQADataset(data.Dataset):
    def __init__(self, args, pretrained="dmis-lab/biobert-large-cased-v1.1-squad", 
                 padding='max_length', truncation=True,
                 question_len=20, 
                 task='classification', mode='train', OD_feature_extractor = None, OD_model = None, OD_image_processor = None,
                 transform=None):
        
        self.padding = padding
        self.truncation = truncation
        self.task = task
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self.data_dir = args.data_dir
        self.dataset_path = os.path.join(self.data_dir, f'{mode}.csv')
        self.image_dir = os.path.join(self.data_dir, mode)
        self.json_path = os.path.join(self.data_dir, 'answer.json')
        print('Reading dataset...')
        with open(self.dataset_path, mode = 'r', encoding='utf8') as csv_file:
            csv_dict_reader = DictReader(csv_file)
            self.entries = list(csv_dict_reader)
        
        self.ans2id = json.load(open(self.json_path, encoding="utf8"))
        self.id2ans = { v: k for k, v in self.ans2id.items() }
        self.num_classes = len(self.ans2id)

        # self.OD_feature_extractor = OD_feature_extractor
        # self.OD_model = OD_model
        # self.OD_image_processor = OD_image_processor
        # self.tokenize(question_len)

    def tokenize(self, max_length=20):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.tokenizer(entry['question'], padding=self.padding, truncation=self.truncation, 
                                    max_length=max_length, return_tensors='pt')
            for key, val in tokens.items():
                tokens[key] = torch.squeeze(val, dim=0)
            # assert_eq(len(tokens['input_ids']), max_length)
            entry['question'] = tokens


    def __getitem__(self, item):
        question = self.entries[item]['question']
        question = ViTokenizer.tokenize(question)
        img_path = os.path.join(self.image_dir, f"{self.entries[item]['img_id']}.jpg")
        answer = self.entries[item]['answer']
        label = self.ans2id[answer]

        assert os.path.exists(img_path), ('{} does not exist'.format(img_path))
        
        img = Image.open(img_path)
        w,h = img.size 
        size = (h, w)

        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np, img_np, img_np), axis=-1)
            img = Image.fromarray(img_np.astype(np.uint8))

        label_placeholder = Image.fromarray(np.zeros(size, dtype=np.uint8))

        sample_t = {'image': img, 'label': label_placeholder}
        if self.transform:
            tensor_img = transforms.ToTensor()(img)
            sample_t = self.transform(tensor_img)
            sample_t = transforms.ToPILImage()(sample_t)

        else:
            sample_t = img

        # OD_inputs = self.OD_feature_extractor(images=img, return_tensors="pt")
        # OD_inputs = OD_inputs.to(self.device)
        # OD_outputs = self.OD_model(**OD_inputs)

        # list_box = torch.permute(OD_outputs['pred_boxes'], (0, 2, 1)).detach() 
        # list_box = OD_outputs['pred_boxes']
        return {
            'question': question,
            'org_image': img,
            'image': sample_t,
            'label': label,
            'answer': answer
        }


    def __len__(self):
        return len(self.entries)


class VTCollator:

    def __init__(self, feature_extractor, tokenizer, question_length=14, store_origin_data=False): 
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.question_length = question_length
        self.store_origin_data = store_origin_data
        # self.PIL2tensor_transforms = transforms.transforms.ToTensor() 



    def __call__(self, batch):
        encodings = {}
        if self.store_origin_data: 
            # encodings['org_image'] = torch.cat([self.PIL2tensor_transforms(x['org_image']) for x in batch])
            encodings['org_image'] = [x['org_image'] for x in batch]       
            # encodings['org_question'] = [x['question'] for x in batch]
            # encodings['answer'] = [x['answer'] for x in batch]
            
        encodings['image'] = self.feature_extractor([x['image'] for x in batch], return_tensors='pt') 
        encodings['question'] = self.tokenizer([x['question'] for x in batch], padding='max_length', 
                                               max_length=self.question_length, truncation=True, return_tensors='pt')
        encodings['label'] = torch.tensor([x['label'] for x in batch])

        # encodings['bboxes'] = [x['list_box'] for x in batch]

        
        

        return encodings

    def filter_real_bbox(self, output_OD):
        boxes = output_OD['boxes']
        filtered_box = []
        for box in boxes:
            xmin,ymin,xmax,ymax = box
            if xmin >=0 and ymin >= 0 and xmax >=0 and ymax >=0:
                if xmax - xmin >= 1:
                    if ymax - ymin >= 1:
                        filtered_box.append(box.detach().numpy())
        return filtered_box

        