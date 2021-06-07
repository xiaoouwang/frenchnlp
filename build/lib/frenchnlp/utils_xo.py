import torch
import pandas as pd
from os import walk
import numpy as np
import json
import pickle
import bz2
import _pickle as cPickle
import importlib
import sys
import transformers as ppb
from transformers import pipeline
import re
from sklearn.metrics import confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

def deft_overlap_ratio(df):
    set1, set2 = set(df.EltCorrections_normalized), set(df.texteRep_normalized)
    total = len(set1)
    inter = len(set1.intersection(set2))
    return inter/total


def deft_tester(df, features, label):
    num_features = len(features)
    if num_features == 1:
        X_train = df[features[0]].to_numpy().reshape(-1, 1)
    else:
        total_features = len(features)
        if total_features == 2:
            values1 = df[features[0]].tolist()
            values2 = df[features[1]].tolist()
            X_train = [[x, y] for x, y in zip(values1, values2)]
    print(len(X_train[0]))
    X_test = X_train
    y_train = df[label].apply(lambda x: round(x, 1)).tolist()
    y_train = [str(x) for x in y_train]
    y_test = y_train
    linear = svm.SVC(kernel='linear', C=1,
                     decision_function_shape='ovo').fit(X_train, y_train)
    rbf = svm.SVC(kernel='rbf', gamma=1, C=1,
                  decision_function_shape='ovo').fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1,
                   decision_function_shape='ovo').fit(X_train, y_train)
    sig = svm.SVC(kernel='sigmoid', C=1,
                  decision_function_shape='ovo').fit(X_train, y_train)
    linear_pred = linear.predict(X_test)
    # print(X_test[:5])
    poly_pred = poly.predict(X_test)
    # print(X_test[:5])
    rbf_pred = rbf.predict(X_test)
    # print(X_test[:5])
    sig_pred = sig.predict(X_test)
    clf = MultinomialNB().fit(X_train, y_train)
    # clf_pred = clf.predict(x)
# print(clf_pred.reshape(-1, 1))
    accuracy_bayes = clf.score(X_test, y_test)
    print(set(linear_pred), set(poly_pred), set(rbf_pred), set(sig_pred))
    # print(X_test[:5])
    # retrieve the accuracy and print it for all 4 kernel functions
    accuracy_lin = linear.score(X_test, y_test)
    accuracy_poly = poly.score(X_test, y_test)
    accuracy_rbf = rbf.score(X_test, y_test)
    accuracy_sig = sig.score(X_test, y_test)
    print("Accuracy Linear Kernel:", accuracy_lin)
    print("Accuracy Polynomial Kernel:", accuracy_poly)
    print("Accuracy Radial Basis Kernel:", accuracy_rbf)
    print("Accuracy Sigmoid Kernel:", accuracy_sig)
    print("Accuracy Naive Bayes:", accuracy_bayes)


def deft_html_cleaning(text):
    # remove tags
    text = re.sub('<[^>]*>', ' ', str(text))
    # remove non-breaking space
    text = text.replace("&nbsp;", " ")
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    # text = text.replace(' ', '')
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    # remove multiples spaces
    return text


def xo_merge_files(path_in, path_out, extension):
    # initiate a empty string
    final_text = ""
    # get all the files with extension js, the walk function gives a tuple
    for (dirpath, dirname, filenames) in walk(path_in):
        for file in filenames:
            # the / is important :D
            my_path = dirpath + "/" + file
            if my_path.endswith('.' + extension):
                with open(my_path) as file:
                    # read the whole file
                    final_text += file.read()
    with open(path_out, "w") as f:
        f.write(final_text)


def xo_import_commons():
    import pandas as pd
    import re


def xo_read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except:
        print("file doesn't exist")


def xo_json2csv(input_path, output_path):
    df = pd.read_json(input_path)
    df.to_csv(output_path, index=False)


def xo_csv2json_nonull(in_path, out_path):
    df = pd.read_csv(in_path)
    xo_write_json_fromdf(out_path, df)
    text = open(out_path, "r").read()
    new_text, n = re.subn(":null,", ': "",', text)
    xo_write_file(out_path, new_text)


def xo_read_lines(path):
    lines = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def xo_write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def xo_write_json_fromdf(path, df):
    df.to_json(path, force_ascii=False, orient="records")


def xo_write_json_fromlist(path, lst):
    """write to json from a list of dicts

    Args:
        path (string): path for the output file
        lst (list): a list of dict
    """

    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(lst, outfile, ensure_ascii=False)


def xo_count_words(s):
    return len(s.split())


def xo_replace_with_mask(sent, index, word):
    sent = sent.split()
    sent[index] = "<mask>"
    masked_sent = " ".join(sent).replace("' ", "'").replace(
        " ,", ",").replace(" .", ".").replace("$$ ", "")
    return masked_sent


def xo_prob_options(obj_camembert, options):
    # print(obj_camembert)
    answers = [(x["token_str"].strip(), x["score"])
               for x in obj_camembert if x["token_str"].strip() in options]
    # print(answers)
    # different number of answers
    if len(answers) == 0:
        return (0, 0, 0, 0)
    elif len(answers) == 1:
        return (answers[0][0], answers[0][1], 0, 0)
    else:
        return (answers[0][0], answers[0][1], answers[1][0], answers[1][1])


def xo_load_data(path, sep=";"):
    # pd_good = pd.read_csv("wino_for_bert_changed.csv", sep=sep)
    pd_good = pd.read_csv(path, sep=sep)
    # pd_good = pd_good.iloc[:, 1:]
    col_names = list(pd_good.columns)
    # add column names to remove the col names
    pd_good = pd.read_csv(path,
                          sep=sep, names=col_names, header=None)
    # pd_good = pd_good.iloc[:, 1:]
    pd_good["response1"] = 0
    pd_good["prob1"] = 0
    pd_good["response2"] = 0
    pd_good["prob2"] = 0
    pd_good = pd_good.iloc[1:, :].reset_index(drop=True)
    return pd_good


def xo_load_cam(cam_model):
    camembert = ppb.CamembertModel.from_pretrained(cam_model)
    return camembert


def xo_fillin(model_str, k):
    task = pipeline('fill-mask', model=model_str, top_k=k)
    return task


def xo_produce_answers(pipeline, col, pd_good):
    for i, row in pd_good.iterrows():
        masked_line = row[col]
        options = row.options.split()
        answers = pipeline(masked_line)
        # pd_good.iloc[i]["response1"], pd_good.iloc[i]["prob1"], pd_good.iloc[i][
        # "response2"], pd_good.iloc[i]["prob2"] = xo_prob_options(answers, options)
        pd_good.loc[i, ["response1"]], pd_good.loc[i, ["prob1"]], pd_good.loc[i, [
            "response2"]], pd_good.loc[i, ["prob2"]] = xo_prob_options(answers, options)
    return pd_good


def xo_produce_answers_js(model, pd_good):
    for i, row in pd_good.iterrows():
        masked_line = row.local_context
        options = [row.correct_answer_local_contexte,
                   row.wrong_answer_local_contexte]
        answers = model.fill_mask(masked_line, topk=20000)
        pd_good.loc[i, ["response1"]], pd_good.loc[i, ["prob1"]], pd_good.loc[i, ["response2"]], pd_good.loc[
            i, ["prob2"]] = xo_prob_options(answers, options)
    return pd_good


def xo_test_single_sent(pipeline, sent):
    return pipeline(sent)


def xo_compute_score(correct_col, pd_good):
    counter_correct = 0
    counter_noresponse = 0
    counter_badresponse = 0
    for i, row in pd_good.iterrows():
        if row[correct_col] == row.response1:
            counter_correct += 1
        elif row.response1 == 0:
            counter_noresponse += 1
        else:
            counter_badresponse += 1
    counter_total = counter_correct + counter_badresponse + counter_noresponse
    data = {"correct": [round(counter_correct, 2)], "no_response": [round(counter_noresponse, 2)], "bad_response": [round(counter_badresponse, 2)], "total_responses": [round(counter_total, 2)], "exactitude": [round(
        (counter_correct / counter_total) * 100, 2)], "qualite": [round((counter_correct / (counter_total - counter_noresponse)) * 100, 2)], "reussite": [round(((counter_correct + counter_noresponse / 2) / counter_total) * 100, 2)]}
    return pd.DataFrame(data=data)


def xo_cleanfrwac_alt(fpath):
    fr_pmi_ancien = pd.read_csv(fpath)
    # delete second and 3rd rows and reset index
    fr_pmi_ancien.drop([0, 1], inplace=True)
    fr_pmi_ancien.reset_index(drop=True, inplace=True)
    # col names lowercase and rename second column nb_npropre and keep names simple
    fr_pmi_ancien.rename(str.lower, axis='columns', inplace=True)
    fr_pmi_ancien.rename(columns={
                         'np': 'nb_npropre', 'r0 lm': 'r0', 'r1 lm': 'r1', 'item': 'schema'}, inplace=True)
    # keep 8 first columns
    fr_pmi_ancien.drop(fr_pmi_ancien.columns[8:], axis=1, inplace=True)
    fr_pmi_ancien.dropna(how='all', inplace=True)
    fr_pmi_ancien['type'].replace(np.nan, "alt", inplace=True)
    fr_pmi_ancien['type'].replace("std", "alt", inplace=True)
    # reorder columns
    fr_pmi_ancien = fr_pmi_ancien[[
        "schema", "type", "nb_npropre", "r0", "r1", "question", "special", "alternate"]]
    fr_pmi_ancien_alt = fr_pmi_ancien[fr_pmi_ancien["type"] == "alt"]
    # clean a non existence row
    fr_pmi_ancien_alt.drop([131], inplace=True)
    return fr_pmi_ancien_alt


def xo_read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

# def xo_read_pickle(path):


def xo_decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def xo_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


def xo_reimport_all(package):
    importlib.reload(sys.modules[package])


def xo_reimport_module(some_module):
    importlib.reload(some_module)


def xo_reload_self():
    importlib.reload(sys.modules["utils_xo"])
