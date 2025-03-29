# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
from transformers import AutoTokenizer
import numpy as np
import ast
import pickle
from absl import app, flags
np.random.seed(20222023)
DELIMIT = '\t'


def cleanse_content(raw_content):
    revised = raw_content.replace('-LRB-', '(')
    revised = revised.replace('-RRB-', ')')
    revised = revised.replace('-LSB-', '[')
    revised = revised.replace('-RSB-', ']')
    revised = revised.replace('-LCB-', '{')
    revised = revised.replace('-RCB-', '}')
    return revised


def process_wikilines(wiki_lines):
    ret_dict = {}
    lines = wiki_lines.split('\n')
    for ln in lines:
        contents = ln.split(DELIMIT)
        if contents[0].isdigit():
            lid = int(contents[0])
            if len(contents) > 1:
                line = cleanse_content(contents[1].strip())
            else:
                line = ''
            ret_dict[lid] = line
        else:
            print(contents)
    return ret_dict


def process_wikifiles(from_file, file_id, root_path, index_dict):
    with open(from_file, 'r', encoding='utf-8') as wikif:
        output_fn = 'wiki-' + str(file_id).zfill(3) + '.json'
        wiki_dict = {}
        with open(root_path + output_fn, 'w', encoding='utf-8') as output:
            for line in wikif:
                onewk = json.loads(line)
                wkid = onewk['id'].strip()
                if wkid:  # not empty
                    content = onewk['lines']
                    if content:
                        assert wkid not in wiki_dict and wkid not in index_dict
                        lines_dict = process_wikilines(content)
                        wiki_dict[wkid] = lines_dict
                        index_dict[wkid] = output_fn
            json.dump(wiki_dict, output)
    return index_dict


def write_line(output_file, query, evidence_list, class_lbl, loc_list, write_loc):
    loclist = loc_list if write_loc else []
    output_line = str(query) + DELIMIT + str(evidence_list) + DELIMIT + str(class_lbl) + DELIMIT + str(loclist)
    output_file.write(output_line + '\n')
    return


def collect_evidences(evid_dict, idx_dict, idx_root):
    evid_list = []
    loc_ids = []
    evid_counter = 0
    for evid in evid_dict:
        if evid in idx_dict:
            lids = evid_dict[evid]
            with open(idx_root + idx_dict[evid], 'r', encoding='utf-8') as conf:
                content_dict = json.load(conf)
                content = content_dict[evid]
                for c in content:
                    conline = content[c]
                    if conline:
                        line_idx = int(c)
                        if line_idx in lids:
                            loc_ids.append(evid_counter)
                        evid_list.append(conline)
                        evid_counter += 1
    return evid_list, loc_ids


def evidence_ids(evidence):
    refined_edict = {}
    raw_edict = {}
    for outer_loop in evidence:
        for inner_loop in outer_loop:
            wikid = inner_loop[2]
            wiki_lid = inner_loop[3]
            if wikid not in raw_edict:
                raw_edict[wikid] = []
            raw_edict[wikid].append(wiki_lid)
    # dedup
    for ev in raw_edict:
        refined_edict[ev] = list(set(raw_edict[ev]))
    return refined_edict


def write_raw(input_file, index_dict, index_root, out_filename, write_loc=False):
    def _abbr(lbl):
        if lbl.startswith('NOT ENOUGH INFO'):
            out_lbl = 'NEI'
        elif lbl.startswith('SUPPORTS'):
            out_lbl = 'SUP'
        elif lbl.startswith('REFUTES'):
            out_lbl = 'REF'
        else:
            out_lbl = None
        return out_lbl

    max_bagsize = 0
    max_evidsize = 0
    with open(input_file, 'r', encoding='utf-8') as input, open(out_filename, 'w', encoding='utf-8') as output:
        for j, line in enumerate(input):
            print(j, max_evidsize, max_bagsize)
            obs = json.loads(line)
            claim = obs['claim']
            label = _abbr(obs['label'])
            if label in ['SUP', 'REF']:
                evidence_dict = evidence_ids(obs['evidence'])
                evids, locs = collect_evidences(evid_dict=evidence_dict, idx_dict=index_dict, idx_root=index_root)
                if not evids:
                    label += '_ERROR'
            else:
                evids, locs = [], []

            claim_sz = len(claim.split())
            if claim_sz > max_evidsize:
                max_evidsize = claim_sz
            if len(evids) > max_bagsize:
                max_bagsize = len(evids)
            for evd in evids:
                evd_sz = len(evd.split())
                if evd_sz > max_evidsize:
                    max_evidsize = evd_sz
            write_line(output, query=claim, evidence_list=evids, class_lbl=label, loc_list=locs, write_loc=write_loc)
    # train: 336 258
    # dev: 169 185
    # test: 212 172
    print(max_bagsize, max_evidsize)


def cleanse_raw(input_filename, tokenizer, evidence_threshold=256, line_threshold=256, shuffled=False):
    def _line2ids(_line, _tokenizer):
        return _tokenizer(_line)['input_ids']
        # _tokens = []
        # _tokens.append('[CLS]')
        # _tokens.extend(_tokenizer.tokenize(_line))
        # _tokens.append('[SEP]')
        # return _tokenizer.convert_tokens_to_ids(_tokens)

    count_p = 0
    count_n = 0
    max_linesize_counter = 0
    obs = []
    with open(input_filename, 'r', encoding='utf-8') as inputf:
        for line in inputf:
            max_linesize = 0
            field = line.split(DELIMIT)
            if not (field[2].startswith('NEI') or field[2].endswith('ERROR')):
                class_label = 1 if field[2].startswith('SUP') else 0
                if class_label:
                    count_p += 1
                else:
                    count_n += 1
                claim_ids = _line2ids(field[0], tokenizer)[:line_threshold]
                if len(claim_ids) > max_linesize:
                    max_linesize = len(claim_ids)
                loc_list = ast.literal_eval(field[3])
                loc_index = [_l for _l in loc_list if _l < evidence_threshold]
                evid_list = ast.literal_eval(field[1])
                if len(evid_list) > evidence_threshold:
                    evidences = evid_list[:evidence_threshold]
                else:
                    evidences = evid_list
                # Austria is a film. 274 [0, 1, 3, 6, 8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 59, 93, 98, 105, 119, 120, 144, 172, 173, 184, 189, 194, 202, 226, 234, 239, 259]
                # The Italian language is spoken in at least one country. 336 [2, 3, 4, 5, 9, 10, 22, 50, 51, 79, 92, 116, 118, 142, 165, 179, 205, 219, 246, 256, 280, 305, 323, 324]
                if len(evidences) == evidence_threshold:
                    print(field[0], len(evidences), loc_index)
                evidence_ids = []
                for evid in evidences:
                    evid_ids = _line2ids(evid, tokenizer)[:line_threshold]
                    if len(evid_ids) > max_linesize:
                        max_linesize = len(evid_ids)
                    evidence_ids.append(evid_ids)
                output_obs = np.array([claim_ids, evidence_ids, class_label, loc_index], dtype=object)
                obs.append(output_obs)
                if max_linesize == line_threshold:
                    max_linesize_counter += 1
                    print(count_p, count_n, max_linesize, field[0])
    if shuffled:
        np.random.shuffle(obs)
    print(count_p, count_n, max_linesize_counter)
    return obs

# 341 121 276 Belgium is comprised of three regions.
# 458 170 321 Three Days of the Condor includes a Swedish actor.
# 765 290 321 Three Days of the Condor features a Swedish actor.
# 895 335 276 Belgium is made up of three regions.
# 2615 975 276 Romelu Lukaku is European.
# 3592 1326 276 Belgium is made up of the Flemish Region and two others.
# 8819 3305 276 Belgium is home to multiple linguistic groups.
# 9393 3514 276 Belgium is made up entirely of the Flemish Region since 2012.
# 10357 3871 276 Belgium is home to a small group of German-speakers.
# 18116 6722 276 Belgium has both Dutch-speaking and French-speaking people.
# Austria is a film. 256 [0, 1, 3, 6, 8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 59, 93, 98, 105, 119, 120, 144, 172, 173, 184, 189, 194, 202, 226, 234, 239]
# 25055 9301 276 Belgium is home to some German-speakers.
# 25517 9463 276 Belgium is made up of Wallonia and two more regions.
# 25801 9586 276 Belgium is mostly Flemish.
# 27857 10334 276 Belgium is comprised of Wallonia, Flanders and the Brussels Capital.
# 27898 10345 276 Belgium is not home to two main linguistic groups.
# 28721 10625 312 Venus Williams plays tennis.
# 29700 11016 276 Belgium is home to two main linguistic groups - French and Dutch.
# The Italian language is spoken in at least one country. 256 [2, 3, 4, 5, 9, 10, 22, 50, 51, 79, 92, 116, 118, 142, 165, 179, 205, 219, 246]
# 34104 12688 276 Belgium is comprised of Wallonia and one other region.
# 35499 13194 276 Belgium is not 41% French-speaking.
# 37592 13967 276 Belgium's population is 59% Flemish.
# 39496 14702 276 Belgium is home to two primary linguistic groups.
# 42031 15623 321 There were multiple actors nominated for awards in Game of Thrones.
# 42248 15718 276 Same-sex marriage is legal in parts of Europe.
# 42568 15826 276 Belgium is comprised of Brussels Capital Region and two other regions.
# 44589 16579 276 Belgium is comprised of Wallonia and two other regions.
# 45357 16887 276 Belgium has multiple languages spoken by its people.
# 46795 17408 276 Belgium does not have both Dutch-speaking and French-speaking people.
# 48587 18104 276 Belgium is home to a small group of French-speakers.
# 49766 18575 276 Belgium is home to a tiny group of German-speakers.
# 51230 19105 276 Belgium is 41% French-speaking.
# 52391 19527 276 Dogstar (band) performed in Europe.
# 58192 21729 276 Belgium is home to two main linguistic groups.
# 60137 22374 312 Venus Williams is a person that plays tennis.
# 64518 23970 321 Three Days of the Condor includes a Swedish actor in the role of the primary antagonist.
# 65088 24202 321 Three Days of the Condor includes only American actors.
# 65248 24247 276 Belgium has both Dutch-speaking and French-speaking people in its capital.
# 65372 24283 276 Belgium is not home to a small group of German-speakers.
# 66864 24852 321 Game of Thrones has an international cast of actors.
# 69954 26071 276 Belgium's population is 41% French-speaking.
# 70751 26368 276 Belgium is almost half French-speaking.
# 72114 26866 276 Belgium has both Dutch-speaking and French-speaking folks.
# 72946 27121 276 Belgium is 59% Flemish.
# 73959 27500 321 Three Days of the Condor includes an actor.
# 74097 27560 276 Belgium is 59% British.
# 78961 29341 276 Belgium is 41.22% French-speaking.


flags.DEFINE_string('root_folder', 'FEVER/', 'Folder/path as root to store output and other files')
flags.DEFINE_string('fever_raw', 'raw/', 'Folder/path under root to store raw data: {train,paper_dev,paper_test}.jsonl and wiki-pages/')
flags.DEFINE_string('temp_folder', 'tmp/', 'Folder/path under root to store temporary files during data prep')
flags.DEFINE_integer('number_datasets', 3,
                     'Number of round of experiement that requires construction of datasets')
flags.DEFINE_integer('number_possamples', 16500,
                     'Number of positive exemplars randomly sampled from raw training data',
                     lower_bound=1, upper_bound=79551)
flags.DEFINE_integer('number_negsamples', 16500,
                     'Number of negative exemplars randomly sampled from raw training data',
                     lower_bound=1, upper_bound=29584)
flags.DEFINE_string('train_prefix', 'train_fever', 'Prefix in the output filename of a train dataset')
flags.DEFINE_string('dev_prefix', 'dev_fever', 'Prefix in the output filename of a dev dataset')
flags.DEFINE_string('test_prefix', 'test_fever', 'Prefix in the output filename of a test dataset')
flags.DEFINE_bool('preprocess', True, 'Whether to run the one-off preprocessing based on the raw data from scratch')
FLAGS = flags.FLAGS


def main(unused_argv):
    target_root = FLAGS.root_folder

    if FLAGS.preprocess:
        source_root = FLAGS.root_folder + FLAGS.fever_raw
        assert os.path.exists(target_root), 'Non-existence of the root folder: ' + target_root
        assert os.path.exists(source_root), 'Non-existence of folder to store raw data: ' + source_root
        temp_folder = FLAGS.root_folder + FLAGS.temp_folder
        os.makedirs(temp_folder, exist_ok=True)
        json_index = temp_folder + 'index.json'

        wiki_root = temp_folder + 'wiki-files/'
        os.makedirs(wiki_root, exist_ok=True)
        # process wiki
        wiki_pages = source_root + 'wiki-pages/'
        wfs = os.listdir(wiki_pages)
        idx_dict = {}
        for i, w in enumerate(wfs):
            idx_dict = process_wikifiles(
                wiki_pages + w,
                file_id=(i+1),
                root_path=wiki_root,
                index_dict=idx_dict
            )

        with open(json_index, 'w', encoding='utf-8') as idxf:
            json.dump(idx_dict, idxf)
        # write json to tsv files
        with open(json_index, 'r', encoding='utf-8') as idxf:
            idx_dict = json.load(idxf)
        print('index loading completed')

        write_raw(
            input_file=source_root + 'train.jsonl',
            index_dict=idx_dict,
            index_root=wiki_root,
            out_filename=temp_folder + 'train_raw.tsv',
            write_loc=True
        )
        write_raw(
            input_file=source_root + 'paper_dev.jsonl',
            index_dict=idx_dict,
            index_root=wiki_root,
            out_filename=temp_folder + 'dev_raw.tsv',
            write_loc=True
        )
        write_raw(
            input_file=source_root + 'paper_test.jsonl',
            index_dict=idx_dict,
            index_root=wiki_root,
            out_filename=temp_folder + 'test_raw.tsv',
            write_loc=True
        )

        # cleanse and save tsv to npy files
        model_id = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=target_root+'base_model')
        # vocab_path = bert_root + 'vocab.txt'
        # tokenizer = FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
        eth = 47  # 256
        lth = 96  # 256
        # 79551(72.9%) 29584(27.1%) 45 over 256-lth + 2 over 256-eth
        # 79551 29584 9584 (8.78%)*1.2~10.5%
        # 79551 29584 8282
        all_obs_train = cleanse_raw(temp_folder + 'train_raw.tsv', tokenizer=tokenizer, shuffled=True,
                                    evidence_threshold=eth, line_threshold=lth)
        with open(target_root + FLAGS.train_prefix + '_all.npy', 'wb') as outputft:
            for l in all_obs_train:
                np.save(outputft, l, allow_pickle=True)
        # 3306 3310 0
        # 3306 3310 406*1.2~7.36%
        # 3306 3310 354
        all_obs_dev = cleanse_raw(temp_folder + 'dev_raw.tsv', tokenizer=tokenizer,
                                  evidence_threshold=eth, line_threshold=lth)
        with open(target_root + FLAGS.dev_prefix + '_all.npy', 'wb') as outputfd:
            for l in all_obs_dev:
                np.save(outputfd, l, allow_pickle=True)
        # 3309 3304 8
        # 3309 3304 424
        # 3309 3304 304
        all_obs_test = cleanse_raw(temp_folder + 'test_raw.tsv', tokenizer=tokenizer,
                                   evidence_threshold=eth, line_threshold=lth)
        with open(target_root + FLAGS.test_prefix + '_all.npy', 'wb') as outputfs:
            for l in all_obs_test:
                np.save(outputfs, l, allow_pickle=True)

    # subsampling from raw training data
    rand_p = np.random.random_sample((79551,))
    rand_n = np.random.random_sample((29584,))
    ratio_p = FLAGS.number_possamples / 79551
    ratio_n = FLAGS.number_negsamples / 29584
    for run in range(1, 1 + FLAGS.number_datasets):
        infile = target_root + FLAGS.train_prefix + '_all.npy'
        outfile = target_root + FLAGS.train_prefix + str(run) + '.npy'
        with open(infile, 'rb') as input_train, open(outfile, 'wb') as output_train:
            try:
                counter_p = 0
                counter_n = 0
                while True:
                    datum = np.load(input_train, allow_pickle=True)
                    if datum[2] > 0:
                        if rand_p[counter_p] < ratio_p:
                            np.save(output_train, datum)
                        counter_p += 1
                    else:
                        if rand_n[counter_n] < ratio_n:
                            np.save(output_train, datum)
                        counter_n += 1
            except (EOFError, OSError, pickle.UnpicklingError):
                print('\nLoading completed')
                input_train.close()
                output_train.close()


if __name__ == '__main__':
    app.run(main)
