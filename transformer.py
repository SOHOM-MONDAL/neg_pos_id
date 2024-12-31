from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Example pipeline for text classification
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Predict sentiment
result = classifier("Hi! I am megha. Today l told about my myself l know that l so boring person but l think l most of time in a day l feeling so exhausted . I know most of people favorite winter l also love . also l hate winter because l most of time in day sleepify and dreaming. I am a average students or you can call commonplace or normal students in my class. Most of person say me that l am a great dancer or decent dancer but l believe that l am skilled dancer dancer is not only leisure interest l like dance.")

# Function to map model labels to custom labels
def change(result):
    if result[0]['label'] == 'LABEL_0':
        result[0]['label'] = "positive"
    elif result[0]['label'] == 'LABEL_1':
        result[0]['label'] = "negative"
    return result

# Apply the custom label mapping
custom_result = change(result)
print(custom_result)
