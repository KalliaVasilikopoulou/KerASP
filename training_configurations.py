import json

def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set


with open("user_scripts/training_configurations.json", "r") as f:
    training_configurations = json.load(f)

training_configurations = json_to_dict(training_configurations)
