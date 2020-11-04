from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

categories = ['D00', 'D10', 'D20', 'D40']

# class that defines and loads the kangaroo dataset


class KangarooDataset(Dataset):
	# load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "D00")
        self.add_class("dataset", 2, "D10")
        self.add_class("dataset", 3, "D20")
        self.add_class("dataset", 4, "D40")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
        files = listdir(images_dir)
        val_count = round(len(files) * 0.1, -1)
        # find all images
        for i, filename in enumerate(files):
            # extract image id
            image_id = i
            # skip bad images
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= len(files)-val_count:
            	continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < len(files)-val_count:
            	continue
            img_path = images_dir + filename
            ann_path = annotations_dir + filename.split('.')[0] + '.xml'
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        labels = []
        boxes = []
        width, height = -1, -1
        for obj in root.findall('.//object'):
            name = obj.find('name').text
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            labels.append(name)
            boxes.append(coors)
            width = int(root.find('.//size/width').text)
            height = int(root.find('.//size/height').text)
        return boxes, labels, width, height
 
	# load the masks for an image
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, labels, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = []
        for i in range(len(boxes)):
            if labels[i] not in categories:
                continue
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(labels[i]))
        return masks, asarray(class_ids, dtype='int32')
 
	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
 
# define a configuration for the model
class KangarooConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = len(categories) + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131
 
# prepare train set
train_set = KangarooDataset()
train_set.load_dataset('train2', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = KangarooDataset()
test_set.load_dataset('train2', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = KangarooConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')