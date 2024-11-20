# %%
from dataset import CusttomDataSnor
import pandas as pd  
from glob import glob  
import librosa.display
import matplotlib.pyplot as plt
import librosa



df = pd.DataFrame()
df["Link"] = glob("Snoring Dataset/*/*.wav")
df["Label"] = df["Link"].apply(lambda x: x.split("/")[-2])

data = CusttomDataSnor(df)

# %% 
data[0]['data'][0].shape

# %%
plt.hist(data[0]['data'][0].reshape(-1))
# %%
librosa.display.specshow(data[0]['data'][0].numpy(), y_axis='mel')
plt.savefig("okela.png")
# %%
librosa.display.specshow(data[0]['data'][1].numpy(), y_axis='mel')
plt.savefig("okela.png")

# %% 
librosa.display.specshow(data[0]['data'][2].numpy(), y_axis='mel')
plt.savefig("okela.png")
# %%

librosa.display.specshow(0.5*(data[0]['data'][2].numpy()+data[0]['data'][1].numpy()), y_axis='mel')