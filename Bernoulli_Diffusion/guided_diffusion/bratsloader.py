import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from scipy import ndimage
from visdom import Visdom
viz = Visdom(port=8850)
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    print('f', f)
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys(), root}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            number=filedict['t1'].split('/')[4]
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            path2 = './data/brats/test_labels/' + str(
                number) + '-label.nii.gz'


            seg=nibabel.load(path2)
            seg=seg.get_fdata()
            image = torch.zeros(5, 256, 256)
            image[:4, 8:-8, 8:-8] = out
            label = torch.tensor(seg[None, ...])
            label[label>=1]=1
            print('label', label.shape)
            image[-1:,8:-8, 8:-8 ] = label
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
            if label.sum()>20:
                weak_label=1

            elif label.sum()==0:
                weak_label=0
                name = os.path.join("./data/brats/val_healthy", str(number) + '.nii.gz')
                print('name', name)
                final_img = nibabel.Nifti1Image(np.array(image), affine=np.eye(4))
                nibabel.save(final_img, name)
                print('saved healthy', str(number))
        else:
            image = torch.zeros(4,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = torch.tensor(out[-1, ...][None, ...])
            print('got until here')
            if label.max()>0:
                weak_label=1


            elif label.sum() == 0:
                weak_label=0
                name = os.path.join("./data/brats/val_healthy", str(number) + '.nii.gz')
                print('name', name)
                final_img = nibabel.Nifti1Image(np.array(image), affine=np.eye(4))
                nibabel.save(final_img, name)
                print('saved healthy', str(number))

            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number )

    def __len__(self):
        return len(self.database)




if __name__ == '__main__':
     ds = BRATSDataset( "./data/brats/testing", test_flag=True)
     g=0; k=0
     dl = torch.utils.data.DataLoader(
         ds,
         batch_size=1,
         shuffle=True)
     for (image, out_dict, weak_label, label, number ) in dl:
     #
           print('proportion', number, image.shape)
     #      viz.image(visualize(image[0, 0, ...]), opts=dict(caption="img input 0"))
     #      viz.image(visualize(image[0, 3, ...]), opts=dict(caption=str(weak_label)))
     #      viz.image(visualize(image[0,4, ...]), opts=dict(caption='seg'))
     #

