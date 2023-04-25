import random

from dataset_downloader import load_data
import openai

train, test = load_data()

test_subset = test.sample(100, random_state=42)

# write to Excel file, one with value and one without

openai.api_key = "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"

for i, row in test_subset.iterrows():
    prompt = f"Analyst rating for {row['stock']} on {row['date'].strftime('%Y-%m-%d')}"
    prompt += ". Please output either '1' or '0' to indicate whether this headline is positive or negative."
    prompt += f"Headline:\n{row['title']}\n"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=2,
    )

    response = response["choices"][0]["text"].strip()

    if response not in ("0", "1"):
        response = random.choice(["0", "1"])
        print("Invalid response, using random value instead", i)
    else:
        print("Valid response", i)

    test_subset.loc[row.name, "gpt-value"] = response

test_subset.to_excel("test_subset.xlsx")

test_subset.to_excel("test_subset_no_value.xlsx", columns=["title", "stock"])
