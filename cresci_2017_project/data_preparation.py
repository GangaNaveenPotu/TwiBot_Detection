import ijson
import re
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@[^\s]+', '', text) # Remove @mentions
    text = re.sub(r'#', '', text) # Remove '#' symbol
    text = re.sub(r'rt[\s]+', '', text) # Remove RT
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
    return text

def process_node_json(node_file, label_file, split_file, output_file):
    
    labels = pd.read_csv(label_file)
    splits = pd.read_csv(split_file)

    
    data = labels.merge(splits, on='id')

    
    user_data = {}
    
    with open(node_file, 'r', encoding='utf-8') as f:
        
        for node in tqdm(ijson.items(f, 'item'), desc="Processing node.json"):
            try:
                user_id = node['id']
                
                
                text = (node.get('description', '') or '') + ' ' + ' '.join([tweet.get('text', '') for tweet in node.get('tweets', [])])

                
                # Derived features (Improvement Plan 4.2)
                followers = node.get('followers_count', 0)
                following = node.get('friends_count', 0)
                tweets = node.get('statuses_count', 0)
                created_at_str = node.get('created_at')

                followers_following_ratio = followers / (following + 1)

                tweets_per_day = 0
                if created_at_str:
                    try:
                        # Example format: 'Sat Mar 10 09:27:13 +0000 2018'
                        created_at = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S +0000 %Y')
                        time_diff = datetime.now() - created_at
                        if time_diff.days > 0:
                            tweets_per_day = tweets / time_diff.days
                    except ValueError:
                        pass # Keep 0 if parsing fails

                user_data[user_id] = {
                    'text': text,
                    'followers': followers,
                    'following': following,
                    'tweets': tweets,
                    'verified': node.get('verified', False),
                    'followers_following_ratio': followers_following_ratio,
                    'tweets_per_day': tweets_per_day
                }
            except (KeyError) as e:
                print(f"Skipping item due to error: {e}")
                continue

    
    processed_data = []
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Combining data"):
        user_id = row['id']
        if user_id in user_data:
            user_info = user_data[user_id]
            processed_data.append({
                'id': user_id,
                'text': clean_text(user_info['text']),
                'followers': user_info['followers'],
                'following': user_info['following'],
                'tweets': user_info['tweets'],
                'verified': user_info['verified'],
                'followers_following_ratio': user_info['followers_following_ratio'],
                'tweets_per_day': user_info['tweets_per_day'],
                'label': row['label'],
                'split': row['split']
            })


    
    df = pd.DataFrame(processed_data)

    # Priority 1: Fix NaN values in 'verified' column (boolean converted to int)
    if 'verified' in df.columns:
        if df['verified'].isnull().any():
            print("NaN values found in 'verified' column and filled with False.")
            df['verified'] = df['verified'].fillna(False)
        df['verified'] = df['verified'].astype(int)

    # Check for NaN and Infinite values (Improvement Plan 1.1)
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Fill NaN values with 0
    if df[numerical_cols].isnull().sum().sum() > 0:
        print("NaN values found and filled with 0 in numerical columns.")
        df[numerical_cols] = df[numerical_cols].fillna(0)
    
    # Replace infinite values with the maximum finite value in each column
    for col in numerical_cols:
        if df[col].isin([float('-inf'), float('inf')]).any():
            print(f"Infinite values found in column '{col}' and replaced with max finite value.")
            max_finite = df[col][~df[col].isin([float('-inf'), float('inf')])].max()
            df[col] = df[col].replace([float('-inf'), float('inf')], max_finite)

    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == '__main__':
    node_file = 'D:/Capstone/cresci-2017/node.json'
    label_file = 'D:/Capstone/cresci-2017/label.csv'
    split_file = 'D:/Capstone/cresci-2017/split.csv'
    output_file = 'D:/Capstone/cresci-2017/processed_data.csv'
    process_node_json(node_file, label_file, split_file, output_file)