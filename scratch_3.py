from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load a pre-trained sentiment analysis model (e.g., FinBERT)
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentence_sentiment(sentence):
  """
  Predicts the sentiment of a given sentence using the pre-trained model.

  Args:
    sentence: The sentence to analyze.

  Returns:
    A tuple containing the predicted sentiment label and the corresponding probability score.
  """
  inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
  outputs = model(**inputs)
  logits = outputs.logits
  predicted_class_id = logits.argmax().item()
  probs = torch.softmax(logits, dim=1)
  predicted_prob = probs[0][predicted_class_id].item()

  labels = ['positive', 'negative', 'neutral']  # Adjust labels based on the model
  predicted_label = labels[predicted_class_id]

  return predicted_label, predicted_prob

def filter_sentences_by_sentiment(paragraph):

  sentences = paragraph.split('. ')
  filtered_sentences = []
  for sentence in sentences:
    predicted_label, prob = get_sentence_sentiment(sentence)
    if (predicted_label == "positive" or predicted_label == "negative") and prob >= 0.6:
      filtered_sentences.append(sentence)
  return filtered_sentences

# Example usage
paragraph = "The stock market experienced a significant decline yesterday. However, analysts predict a strong recovery in the coming weeks. The company's recent earnings report was disappointing. prof chamath has been appointed as the lecturer in university of Moratuwa"
filtered_sentences = filter_sentences_by_sentiment(paragraph)
print(filtered_sentences)