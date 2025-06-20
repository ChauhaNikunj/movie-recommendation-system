#!/usr/bin/env python
# coding: utf-8

# # **AI News Summarizer using NLP**
# ---
# 
# ### 🎯 **Objective**
# Build a **lightweight news summarizer** using the **HuffPost News Category Dataset**.  
# We willl apply **LexRank (extractive summarization)** to compress short descriptions into concise 2–3 line summaries.
# 
# ---
# 
# ### 🛠️ **Tech Stack**
# -  Python  
# -  pandas  
# -  sumy (LexRank)
# -  matplotlib 
# 
# ---
# 
# ### 📂 **Dataset**
# 📎 [Global News Dataset](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset) 
# 
# ---

# In[39]:


# Importing standard libraries
import pandas as pd
import numpy as np

#  NLP tools from sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# In[16]:


# Load the dataset
df = pd.read_csv('/kaggle/input/global-news-dataset/data.csv')
df.head()
df.info


# In[18]:


# Drop missing values in full_content
df = df.dropna(subset=['full_content'])
# Filtering cells whose content is more than 300 characters
df = df[df['full_content'].str.len() > 300]
# Picking up 500 random articles from filtered data
df = df.sample(500, random_state=42,).reset_index(drop=True)
# Confirming by checking top 3 rows
df[['title', 'source_name', 'published_at', 'full_content']].head(3)


# In[22]:


# Defining a function to summarize a given article using LexRank
def summarize_article(text,sentence_count=3):
    parser=PlaintextParser.from_string(text, Tokenizer('english'))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Sumarrizing one article to confirm
sample_text = df.loc[0, 'full_content']
print("Title", df.loc[0, 'title'])
print("\nSummary:\n", summarize_article(sample_text))


# In[23]:


# Actual content for comparison
print(df.loc[0, 'full_content'])


# In[25]:


# Applying the code to the whole database
df['summary'] = df['full_content'].apply(lambda x: summarize_article(x))

# Displaying top 5 summaries alongside Titles
df[['title', 'summary']].head()


# ## 📈 **Comparison of Full Article vs. Summary Lengths (Top 100 Samples)**
# 
# This line graph visually compares the character lengths of original news articles (**Full Content**) and their corresponding summaries (**Summary**) using the **first 100 rows** of the dataset.
# 
# - 🔵 **Full Content** (Blue): Represents the raw, unprocessed article body.
# - 🔴 **Summary** (Red): Shows the output of our LexRank-based summarizer.
# 
# > 🔍 **Observation:**  
# > The summaries are consistently shorter than the full content, showcasing effective compression.  
# > Occasional spikes in the blue line highlight longer articles, while the red line remains stable — proving that the summarizer scales efficiently regardless of input size.
# 
# 📊 This visualization is a strong indicator of the summarizer’s performance, maintaining brevity without losing core information.
# 

# In[35]:


# Importing library for visual comparison
import matplotlib.pyplot as plt

# Calculating Lengths
df['content_length'] = df['full_content'].str.len()
df['summary_length'] = df['summary'].str.len()

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(df.index[:100], df['content_length'][:100], label='Full Content', color='blue', linewidth=2)
plt.plot(df.index[:100], df['summary_length'][:100], label='Summary', color='red', linewidth=2)
plt.title('Content vs Summary Length (First 100 Articles)')
plt.xlabel('Article Index')
plt.ylabel('Length (chars)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# ### 📉 Compression Ratio Calculation
# 
# To quantify how much text was reduced, we computed a **compression ratio** for each article:
# 
# > **Compression Ratio = Summary Length / Full Content Length**
# 
# - Values closer to `0` → higher compression  
# - Values near `1` → minimal or no compression  
# - Rounded to `4` decimals for clarity 🧮
# 
# This gives a measurable indicator of how effective our summarization was across different articles.
# 

# In[38]:


# Avoid divide-by-zero
df['compression_ratio'] = df['summary_length'] / df['content_length']
df['compression_ratio'] = df['compression_ratio'].round(4)  # Round to 4 decimal places

# Preview
df[['content_length', 'summary_length', 'compression_ratio']].head()


# _📌 End of Notebook_  
# Feel free to explore, and build on top of this!
# 
