
import ijson
import pandas as pd
from tqdm import tqdm
import os
import re

def clean_text(text):
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text)
    # Remove mentions
    text = re.sub(r'@\w+', 'USER', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Lowercase
    text = text.lower()
    # Remove excessive special characters, keeping alphanumeric, spaces and common punctuation
    text = re.sub(r'[^a-z0-9\s.,;!?-]', '', text) 
    return text

def process_twibot_file(file_path, split):
    print(f"Processing {file_path} for split '{split}'...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        parser = ijson.items(f, 'item')
        for user_obj in tqdm(parser, desc=f"Parsing {os.path.basename(file_path)}"):
            try:
                user_id = user_obj.get('ID')
                label = user_obj.get('label')
                
                profile = user_obj.get('profile', {})
                if not profile:
                    continue
                
                followers = profile.get('followers_count')
                following = profile.get('friends_count')
                listed_count = profile.get('listed_count') # Added listed_count
                tweet_count = profile.get('statuses_count')
                verified_str = profile.get('verified', 'False')

                if isinstance(verified_str, str):
                    verified = 1 if verified_str.strip().lower() == 'true' else 0
                else:
                    verified = 1 if verified_str else 0
                
                description = profile.get('description', '') or ""
                
                tweets = user_obj.get('tweet', [])
                if tweets and isinstance(tweets, list):
                    tweet_text = ' '.join(t for t in tweets if t)
                else:
                    tweet_text = ""

                combined_text = clean_text((description + ' ' + tweet_text).strip()) # Apply text cleaning

                data.append({
                    'id': user_id,
                    'text': combined_text,
                    'followers': followers,
                    'following': following,
                    'listed_count': listed_count, # Added listed_count
                    'tweets': tweet_count,
                    'verified': verified,
                    'label': label,
                    'split': split
                })
            except (ijson.JSONError, ValueError) as e:
                print(f"Skipping an item in {os.path.basename(file_path)} due to parsing error: {e}")
                continue
    return data

def main():
    base_dir = 'twibot_20_project/Twibot-20 kaggle dataset'
    train_file = os.path.join(base_dir, 'train.json')
    dev_file = os.path.join(base_dir, 'dev.json')
    test_file = os.path.join(base_dir, 'test.json')

    train_data = process_twibot_file(train_file, 'train')
    val_data = process_twibot_file(dev_file, 'val')
    test_data = process_twibot_file(test_file, 'test')

    all_data = train_data + val_data + test_data
    df = pd.DataFrame(all_data)

    for col in ['followers', 'following', 'listed_count', 'tweets']: # Added listed_count
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

    df['followers'] = df['followers'].fillna(df['followers'].median())
    df['following'] = df['following'].fillna(df['following'].median())
    df['listed_count'] = df['listed_count'].fillna(df['listed_count'].median()) # Fill NaN for listed_count
    df['tweets'] = df['tweets'].fillna(df['tweets'].median())
    df['verified'] = df['verified'].fillna(0) 

    for col in ['followers', 'following', 'listed_count', 'tweets', 'verified', 'label']: # Added listed_count
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    output_path = 'twibot_20_project/twibot20_processed_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully processed all files and saved to {output_path}")
    print("\nData Head:")
    print(df.head())
    print("\nData Info:")
    df.info()
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    print("\nSplit Distribution:")
    print(df['split'].value_counts())

if __name__ == '__main__':
    main()
