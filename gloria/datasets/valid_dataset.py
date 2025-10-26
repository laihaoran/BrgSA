from torch.utils.data import DataLoader, Dataset
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, cfg, split="valid", transform=transform):
        self.cfg = cfg
        self.image_paths = cfg.image_paths
        self.mask_paths = cfg.mask_paths
        self.transform = transform
        self.disease_to_organ_map = {
        "There is Medical material": "outline",
        "There is Arterial wall calcification": "blood_vessels",
        "There is Cardiomegaly": "heart",
        "There is Pericardial effusion": "heart",
        "There is Coronary artery wall calcification": "blood_vessels",
        "There is Hiatal hernia": "outline",
        "There is Lymphadenopathy": "outline",
        "There is Emphysema": "lung",
        "There is Atelectasis": "lung",
        "There is Lung nodule": "lung",
        "There is Lung opacity": "lung",
        "There is Pulmonary fibrotic sequela": "lung",
        "There is Pleural effusion": "lung",
        "There is Mosaic attenuation pattern": "lung",
        "There is Peribronchial thickening": "lung",
        "There is Consolidation": "lung",
        "There is Bronchiectasis": "lung",
        "There is Interlobular septal thickening": "lung"
        }
        self.prompt = [value[0] for key, value in cls_prompt.items()]
        self.organ = self.map_disease_to_organ(self.prompt)
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        processed_img, processed_mask = self.get_imgs(img_path, mask_path, self.transform)
        organ_dict = self.organ_mapping(processed_mask.squeeze(0), self.organ)
        return processed_img, organ_dict, self.organ
    
    def map_disease_to_organ(self, disease_list):
        return list(map(lambda disease: self.disease_to_organ_map.get(disease), disease_list))

    def organ_mapping(self, mask, organ_list):
            organ_mask_label = {
                "spleen": [1],
                "kidney": [2, 3, 23, 24],
                "gallbladder": [4],
                "liver": [5],
                "stomach": [6],
                "pancreas": [7],
                "adrenal_gland": [8, 9],
                "lung": [10, 11, 12, 13, 14],
                "esophagus": [15],
                "trachea": [16],
                "thyroid_gland": [17],
                "small_bowel": [18, 19],
                "colon": [20],
                "urinary_bladder": [21],
                "prostate": [22],
                "vertebrae": [
                    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                    45, 46, 47, 48, 49, 50
                ],
                "heart": [51, 61],
                "blood_vessels": [
                    52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 
                    63, 64, 65, 66, 67, 68
                ],
                "humerus": [69, 70],
                "scapula": [71, 72],
                "clavicula": [73, 74],
                "femur": [75, 76],
                "hip": [77, 78],
                "spinal_cord": [79],
                "gluteus": [
                    80, 81, 82, 83, 84, 85
                ],
                "autochthon": [86, 87],
                "iliopsoas": [88, 89],
                "brain": [90],
                "skull": [91],
                "ribs_left": [
                    92, 93, 94, 95, 96, 97, 98, 99, 100, 
                    101, 102, 103
                ],
                "ribs_right": [
                    104, 105, 106, 107, 108, 109, 110, 111, 
                    112, 113, 114, 115
                ],
                "sternum": [116],
                "costal_cartilages": [117],
                "outline": [118]
            }
            
            result = []
            patch_shape = (16, 16, 8)
            mask_shape = mask.shape
            
            # Calculate the number of patches along each dimension
            num_patches = (
                mask_shape[0] // patch_shape[0],
                mask_shape[1] // patch_shape[1],
                mask_shape[2] // patch_shape[2]
            )
            
            organ_indices = {organ: [] for organ in organ_list}
            unique_organs = list(organ_indices.keys())  #correct
            sequence_index = 0
            
            for z in range(num_patches[2]):
                for y in range(num_patches[1]):
                    for x in range(num_patches[0]):
                        # Extract the patch
                        patch = mask[
                            x * patch_shape[0]:(x + 1) * patch_shape[0],
                            y * patch_shape[1]:(y + 1) * patch_shape[1],
                            z * patch_shape[2]:(z + 1) * patch_shape[2]
                        ]
                        
                        unique_patch_values = np.unique(patch)
                        
                        for organ in unique_organs:    # correct
                            labels = organ_mask_label.get(organ, [])
                            # Check if any of the labels for the organ are present in the patch
                            if any(label in unique_patch_values for label in labels):
                                organ_indices[organ].append(sequence_index)
                        
                        sequence_index += 1
            # result = [organ_indices[organ] for organ in organ_list]
            return organ_indices

    def get_imgs(self, img_path, mask_path, transform=None):

        nii_img = nib.load(str(img_path))
        img = nii_img.get_fdata()

        mask = nib.load(str(mask_path))
        mask = mask.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
            mask = resize(mask, self.cfg.data.image.imsize, mode='reflect', order=0, anti_aliasing=True)


        if transform is not None:
            img = {"image": img, "mask": mask}
            transformed_data = transform(img)
        return transformed_data["image"], transformed_data["mask"]

def collate_fn(batch):
    process_images = []
    all_organ_dict = []
    all_organ_map = []
    for b_idx, b in enumerate(batch):
        img, organ_dict, orgam_map = b
        process_images.append(img)
        all_organ_dict.append(organ_dict)
        all_organ_map.append(orgam_map)
    process_images = torch.stack(process_images)
      # Create return dictionary
    return_dict = {
        "images": process_images,
        "organ_dict": all_organ_dict,
        "organ_map": all_organ_map
    }

    return return_dict