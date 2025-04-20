# pkg imports
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler


NON_BIOMED_SRs = {
    "Hall_2012",
    "Radjenovic_2013",
    "Sep_2021",
    "Smid_2020"
}

def get_synergy_data(sr_ids: list[str] | None) -> pd.DataFrame:
    """
    Returns the subset of SYNERGY SRs that form the development set 
    for pipeline design and iteration.

    Keyword arguments:
    sr_ids -- list of systematic reviews to include by their id
    """
    # our synergy datasets' paths
    synergy_dataset_path = "./data/synergy_dataset/SYNERGY_combined_data.pkl"
    synergy_titles_path = "./data/synergy_dataset/SYNERGY_review_titles.csv"

    # load our datasets
    syn_df = pd.read_pickle(synergy_dataset_path)
    syn_titles_df = pd.read_csv(synergy_titles_path, delimiter='#', index_col='SR_id')
    # a dict to map a citation_id to their title
    citation2title = syn_titles_df.to_dict(orient='dict')['title']

    if cols:
        # only keep the systematic reviews that were asked for
        filtered_df = syn_df[syn_df['SR_id'].isin(sr_ids)]
        
    # insert the title of the systematic review in the DataFrame
    filtered_df.insert(2, 'SR_title', syn_df['SR_id'].apply(lambda id: citation2title[id]))
    
    # get indexes of rows with an empty title or abstract
    na_idxs = filtered_df[filtered_df[['title','abstract']].isna().any(axis=1)].index
    
    # remove the rows with empty titles or abstracts
    filtered_df = filtered_df.drop(na_idxs, axis='index')
    
    # create a column that maps 'label_included' to a boolean 'relevant' column
    filtered_df['relevant'] = filtered_df['label_included'].apply(bool)

    return filtered_df

def create_batched_devset(df: pd.DataFrame,
                          rus: RandomUnderSampler) -> pd.DataFrame:
    """
    Creates a development set batched by systematic review.

    Keyword arguments:
    df  -- the dataframe of systematic review citations to batch
    rus -- the initialised RandomUnderSampler that will perform
           the undersampling
    """
    
    # initialise empty dictionary
    devset = {}

    # iterate through citations grouped by systematic review
    for sr_title, group in df.groupby(by='SR_title'):
        sr_id = group['SR_id'].iloc[0]
    
        Xs, Ys  = rus_dataset(group, ['title', 'abstract'], 'relevant', rus)

        devset[sr_id] = [sr_title]
        devset[sr_id].append([dspy.Example(citation_title=x[0], citation_abstract=x[1], relevant=y)\
                          .with_inputs('citation_title', 'citation_abstract')
                          for x, y in zip(Xs, Ys)])
    return devset
    
def rus_dataset(df: pd.DataFrame,
                Xcols: list[str],
                Ycol: list[str],
                rus: RandomUnderSampler) -> tuple[np.array, np.array]:
    """
    Performs a randomised under sample of a dataframe,
    returning a training/testing dataset.

    Keyword arguments:
    df    -- The dataframe to randomly undersample from
    Xcols -- The columns of the dataframe that will 
             form the inputs
    Ycol  -- The column of the dataframe that will 
             form the label
    rus   -- The inialised RandomUnderSampler that will
             perform the undersampling
    """
    Xs = np.array(df[Xcols])
    Ys = np.array(df[Ycol])
    X_resampled, Y_resampled = rus.fit_resample(Xs, Ys)
    return  X_resampled, Y_resampled
    
        

    