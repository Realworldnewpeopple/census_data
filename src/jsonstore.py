
import json
import os


def json_update(model, valscore, model_name):
    check = os.path.exists("./score/acuuracy_val.json")
    if check:
        pass
    else:
        data = {}
        with open('./score/acuuracy_val.json', 'w') as outfile:
            json.dump(data, outfile)

    with open('./score/acuuracy_val.json', "r+") as jsonFile:
        new_data = json.load(jsonFile)
        val = valscore
        new_data[model_name] = [model, val.tolist()]
        jsonFile.seek(0)  # rewind
        json.dump(new_data, jsonFile)
        jsonFile.truncate()
