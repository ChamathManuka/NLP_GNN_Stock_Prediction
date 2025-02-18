import matplotlib.pyplot as plt
import numpy as np



from transformers import AutoModelForSequenceClassification, AutoTokenizer
#from transformers import HfApi

model_name = "allenai/specter"  # Use this model as LLaMA is not available
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assuming your news data is a list of words (preprocessed using stemming, lemmatization)
news_data = "['approxim', 'complaint', 'lodg', 'annual', 'involv', 'health', 'staff', 'includ', 'doctor', 'health', 'minist', 'ramesh', 'pathirana', 'revealedhealth', 'ministri', 'receiv', 'complaint', 'regard', 'medic', 'neglig']"
news_data1 = "['polic', 'initi', 'investig', 'shoot', 'incid', 'weligama', 'polic', 'sergeant', 'wa', 'kill', 'sub', 'inspector', 'colombo', 'crime', 'divis', 'ccd', 'wa', 'injur', 'went', 'apprehend', 'group', 'allegedli', 'involv', 'drug', 'trade', 'associ', 'nadun', 'chinthaka', 'alia', 'harak', 'kata', 'current', 'custodi', 'crimin', 'investig', 'depart', 'cid', 'shoot', 'erupt', 'weligama', 'ccd', 'pursu', 'suspect', 'drug', 'traffick']"

encoded_inputs = tokenizer(news_data1, return_tensors="pt", padding=True)
vector = model.bert(**encoded_inputs).last_hidden_state[:, 0, :]
vector = vector.detach().numpy()
print(vector)
print(vector.shape)
# Heatmap
plt.imshow(vector, cmap='hot', interpolation='nearest')
plt.show()

# Bar chart
plt.bar(range(768), vector[0])
plt.show()

# Scatter plot
plt.scatter(range(768), vector[0])
plt.show()

# RGB image
rgb_image = np.zeros((256, 256, 3))
rgb_image[:, :, 0] = vector[0, :256]
rgb_image[:, :, 1] = vector[0, 256:512]
rgb_image[:, :, 2] = vector[0, 512:]
plt.imshow(rgb_image)
plt.show()