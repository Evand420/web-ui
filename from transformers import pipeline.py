from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline("text-classification", model="./fine_tuned_model")

# Test the model
result = classifier("This movie was fantastic! I loved it.")
print(result)