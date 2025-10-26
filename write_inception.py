import pandas as pd
import os

path = "/data4/haoranlai/Dataset/INSPECT/test_list.csv"

df = pd.read_csv(path)

filename = df['impression_id'].tolist()

filename = [os.path.join("/data4/haoranlai/Dataset/INSPECT/CTPA_test/", name + '.nii.gz') for name in filename]

# map filename to Path
newdf = pd.DataFrame({'Path': filename})

# save
newdf.to_csv("/data2/haoranlai/Project/gloria/Dataset/INSPECT_image.csv", index=False)
