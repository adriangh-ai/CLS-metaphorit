import pandas as pd

from datasets import Dataset
from datasets import DatasetDict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


def convert_to_dataframe(df_idx:pd.DataFrame, df_corpus:pd.DataFrame) -> pd.DataFrame:
    """
    TODO
    """
    _df = df_corpus.set_index('song_id')
    df = pd.DataFrame()
    df['text1']=_df.loc[df_idx.id1].text.reset_index(drop=['song_id'])
    df['text2']=_df.loc[df_idx.id2].text.reset_index(drop=['song_id'])
    df.reset_index(inplace=True, drop=['index'])
    df['id1'] = df_idx.id1
    df['id2'] = df_idx.id2
    df['split'] = df_idx.split
    df['labels'] = df_idx.sim_rating
    
    return df

def corpus_to_hf_dataset(dataframe:pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    dataset = DatasetDict()
    df = dataframe.copy().sample(frac=1, random_state=42)
    
    le = LabelEncoder()
    df.labels = le.fit_transform(df.labels)

    dataset['train'] = Dataset.from_pandas(df[df.split=='train'])
    dataset['validation'] = Dataset.from_pandas(df[df.split=='dev'])
    dataset['test'] = Dataset.from_pandas(df[df.split == 'test'])

  

    remove_columns = [x for x in df.columns.to_list() if x!='labels'] + ['__index_level_0__']
    df = dataset.map(lambda x: tokenizer(x['Sentence'], truncation=True),
                     batched=True, 
                     remove_columns=remove_columns)
    return df