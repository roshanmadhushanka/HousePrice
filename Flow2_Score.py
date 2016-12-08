import pandas as pd

def get_scored_features():
    score = {}
    for i in range(10):
        data = pd.read_csv('varimp/varimp' + str(i) + '.csv')
        column_name_list = list(data['variable'])
        for j in range(len(column_name_list)):
            if score.has_key(column_name_list[j]):
                score[column_name_list[j]] += j
            else:
                score[column_name_list[j]] = j

    sorted_list = sorted(score.items(), key=lambda x: x[1])
    columns = {}
    for value in sorted_list:
        arr = value[0].split(".")
        if columns.has_key(arr[0]):
            continue
        else:
            columns[arr[0]] = score[value[0]]

    sorted_list = sorted(columns.items(), key=lambda x: x[1])
    return sorted_list

features = get_scored_features()

selected_features = []
for feature in features[:20]:
    selected_features.append(feature[0])

print selected_features