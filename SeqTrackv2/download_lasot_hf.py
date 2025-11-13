from datasets import load_dataset

# Load the LaSOT dataset (this downloads only metadata and samples on demand)
dataset = load_dataset("lasot/lasot", split="train")

# View available class names
classes = dataset.unique("category")
print("Available classes:", classes)

# Select two arbitrary ones (e.g., first two)
class1, class2 = classes[:2]
print("Selected classes:", class1, class2)

# Count samples per selected class
count1 = sum(1 for x in dataset if x["category"] == class1)
count2 = sum(1 for x in dataset if x["category"] == class2)

print(f"{class1} samples:", count1)
print(f"{class2} samples:", count2)
