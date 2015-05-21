import pandas as pd
from numpy import newaxis
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

def read_shot_file(fname):
    raw_shot_data = pd.read_csv(fname)
    raw_shot_data.columns = [col.lower() for col in raw_shot_data.columns]
    raw_shot_data.shot_result = raw_shot_data.shot_result == "made"
    print(raw_shot_data.columns)
    return(raw_shot_data)

def print_descriptive_stats(shots_df):
    #descriptive stats
    print("Number of closest defenders: %d"%(shots_df.closest_defender.unique().size))
    print("Number of shooters: %d"%(shots_df.player_name.unique().size))
    print("Fraction of shots made: %f"%(shots_df.shot_result.mean()))
    return

def make_modeling_data(raw_shot_data):
    my_one_hot = OneHotEncoder()
    shooter = raw_shot_data.player_id.values[:, newaxis]
    shooter_as_one_hot = my_one_hot.fit_transform(shooter)

    for_model = pd.DataFrame(data=shooter_as_one_hot.todense(), 
                            columns=my_one_hot.active_features_)
    for_model['shot_dist'] = raw_shot_data.shot_dist
    for_model['close_def_dist'] = raw_shot_data.close_def_dist
    return(for_model, raw_shot_data.shot_result)

def get_model_accuracy(my_model, x, y):
    my_model.fit(X=x, y=y)
    return(my_model.score(X=x, y=y))

if __name__ == "__main__":
    raw_fname = "../data/shot_logs.csv"
    raw_shot_data = read_shot_file(raw_fname)
    print_descriptive_stats(raw_shot_data)
    x, y = make_modeling_data(raw_shot_data)
    model = LogisticRegression()
    print("Accuracy: %f"%(get_model_accuracy(model, x, y)))