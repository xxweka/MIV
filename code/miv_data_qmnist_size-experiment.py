# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import ast
import numpy as np
import os
import json
from absl import app, flags
DELIMIT = '\t'


def write_line(output_file, dig_tuple, anc_tplst, class_lbl):
    anc_files, anc_digit = zip(*anc_tplst)
    loclist = [i for i, c in enumerate(anc_digit) if c == dig_tuple[1]]
    output_line = dig_tuple[0] + DELIMIT + str(list(anc_files)) + DELIMIT + class_lbl + DELIMIT + str(loclist)
    output_file.write(output_line + '\n')


def write_type2pairs(outfile, pair1, pair1wid, pair2map, class_label, maxna_per_digit):
    fn, dg = pair1
    pair2list = []
    for k in sorted(pair2map):
        fn_list = pair2map[k]
        anc_dig = min(maxna_per_digit, len(fn_list)) if maxna_per_digit else len(fn_list)
        sample_size = np.random.choice(range(1, anc_dig + 1))
        chosen_ancs = np.random.choice(fn_list, size=sample_size, replace=False)
        pair2list.extend([(a, k) for a in chosen_ancs])
    write_line(outfile, dig_tuple=(fn, pair1wid), anc_tplst=pair2list, class_lbl=class_label)
    return len(pair2list)


def write_file(json_filename, out_filename, root_path='', maxna_per_digit=None, num_rounds=1, ptype=2,
               add_nwriters=3, base_nwriters=3):
    assert ptype in [2]
    with open(json_filename, 'r') as jf:
        ids_dict = json.load(jf)
        wrtids = list(ids_dict.keys())
        digimg = {}
        for wid in wrtids:
            digimg[wid] = []
            for d in ids_dict[wid]:
                # first select digimg, then select a fixed set of ancimg for each wid
                fn_list = [root_path+fl for fl in ids_dict[wid][d]]
                num_imgs = min(num_rounds, len(fn_list))
                chosen_digits = np.random.choice(fn_list, size=num_imgs, replace=False).tolist()
                digimg[wid].extend([(c, d) for c in chosen_digits])
                ids_dict[wid][d] = [f for f in fn_list if f not in chosen_digits]

        with open(out_filename, 'w') as outf:
            linecount = 0
            max_na = 0
            for w in wrtids:
                for dt in digimg[w]:
                    f, d = dt
                    other_wids = [wr for wr in wrtids if (wr!=w and d in ids_dict[wr] and len(ids_dict[wr][d])>0)]
                    # print(len(other_wids))
                    # 1
                    if len(ids_dict[w][d]) > 0:
                        writer_map1 = {w: ids_dict[w][d]}
                        other_w = np.random.choice(other_wids, size=(np.random.choice(add_nwriters) + base_nwriters - 1), replace=False)
                        writer_map1.update({owid: ids_dict[owid][d] for owid in other_w})
                        linesize1 = write_type2pairs(outf, pair1=dt, pair1wid=w, pair2map=writer_map1, class_label='1', maxna_per_digit=maxna_per_digit)
                    else:
                        linesize1 = 0
                    # 0
                    all_w = np.random.choice(other_wids, size=(np.random.choice(add_nwriters) + base_nwriters), replace=False)  # +3
                    writer_map0 = {awid: ids_dict[awid][d] for awid in all_w}
                    linesize0 = write_type2pairs(outf, pair1=dt, pair1wid=w, pair2map=writer_map0, class_label='0', maxna_per_digit=maxna_per_digit)
                    linecount += 2
                    if max(linesize0, linesize1) > max_na:
                        max_na = max(linesize0, linesize1)
            # print(linecount, ': ', max_na, '; total # writers =', len(wrtids))


def build_dict(trainfiles):
    line_dict = {}
    for trainfile in trainfiles:
        with open(trainfile, 'r') as tf:
            for line in tf:
                assert line.endswith('\n')
                entries = line.strip().split(DELIMIT)
                bag_size = len(ast.literal_eval(entries[1]))
                if bag_size not in line_dict:
                    line_dict[bag_size] = []
                line_dict[bag_size].append(line)
    return line_dict


def print_stats(linelist):
    bagsizes = []
    line_dict = {}
    for line in linelist:
        assert line.endswith('\n')
        entries = line.strip().split(DELIMIT)
        bag_size = len(ast.literal_eval(entries[1]))
        bagsizes.append(bag_size)
        if bag_size not in line_dict:
            line_dict[bag_size] = []
        line_dict[bag_size].append(line)
    print(len(linelist), ': mean_bagsizes =', np.mean(bagsizes), 'var_bagsizes =', np.var(bagsizes),
          'max_bagsizes =', np.max(bagsizes), 'min_bagsizes =', np.min(bagsizes))


def template(mean, var, sample_size):
    var_mean = var / mean
    p = 1. - var_mean
    temp = np.random.binomial(round(mean / p), p, sample_size)
    dist = {}
    for t in temp:
        if t not in dist:
            dist[t] = 0
        dist[t] += 1
    # print(np.mean(temp), np.var(temp), dist)
    return dist


flags.DEFINE_string('root_folder', 'qmnist/', 'Folder/path as root to store output and other files')
flags.DEFINE_string('temp_folder', 'tmp/', 'Folder/path under root that stores writer-ID json files')
flags.DEFINE_string('image_folder60k', 'train60k/', 'Folder/path under root to store images from MNIST-train60k')
flags.DEFINE_integer('number_datasets', 3,
                     'Number of round of experiement that requires construction of datasets')
flags.DEFINE_string('train_prefix', 'train_qmnist', 'Prefix in the output filename of a train dataset')
flags.DEFINE_integer('mean_bagsize', 10, 'Average number of instances per target bag')
flags.DEFINE_integer('number_exemplars', 8000, 'Number of exemplars in a train dataset')
FLAGS = flags.FLAGS


def main(unused_argv):
    # prelim steps: collect writer IDs to construct train dataset (of all possible different bag sizes)
    # print(os.getcwd())
    temp_folder = FLAGS.root_folder + FLAGS.temp_folder
    ptype = 2
    raw_fns = [
        temp_folder + 'raw0qmnist.tsv',
        temp_folder + 'raw1qmnist.tsv',
        temp_folder + 'raw2qmnist.tsv',
        temp_folder + 'raw3qmnist.tsv'
    ]
    train_json = temp_folder + 'train60k_writerid.json'
    nr = 5
    write_file(json_filename=train_json,
               out_filename=raw_fns[0],
               root_path=FLAGS.image_folder60k,
               maxna_per_digit=15,
               base_nwriters=3,
               add_nwriters=3,
               num_rounds=nr,
               ptype=ptype)
    write_file(json_filename=train_json,
               out_filename=raw_fns[1],
               root_path=FLAGS.image_folder60k,
               maxna_per_digit=10,
               base_nwriters=6,
               add_nwriters=3,
               num_rounds=nr,
               ptype=ptype)
    write_file(json_filename=train_json,
               out_filename=raw_fns[2],
               root_path=FLAGS.image_folder60k,
               maxna_per_digit=10,
               base_nwriters=9,
               add_nwriters=6,
               num_rounds=nr,
               ptype=ptype)
    write_file(json_filename=train_json,
               out_filename=raw_fns[3],
               root_path=FLAGS.image_folder60k,
               maxna_per_digit=5,
               base_nwriters=15,
               add_nwriters=6,
               num_rounds=nr,
               ptype=ptype)
    # template(50, 10, 24000)
    # template(20, 4, 24000)
    # template(10, 2, 32000)

    nexemplars = FLAGS.number_exemplars  # 32000  # 8000  # 24000  # 16000  #
    train_fns = [FLAGS.root_folder + FLAGS.train_prefix + str(ds) + '.tsv'
                 for ds in range(1, FLAGS.number_datasets + 1)]
    fns = raw_fns + train_fns if nexemplars > 10000 else raw_fns
    if nexemplars > 24000:
        add_fns = temp_folder + 'raw10qmnist.tsv'
        write_file(
            json_filename=train_json,
            out_filename=add_fns,
            root_path=FLAGS.image_folder60k,
            maxna_per_digit=3,
            base_nwriters=5,
            add_nwriters=1,
            num_rounds=1,
            ptype=ptype
        )
        fns += [add_fns]
    bag_dict = build_dict(fns)
    lines = []
    for k in bag_dict:
        print(k, ': ', len(bag_dict[k]))
        lines.extend(bag_dict[k])
    # print_stats(lines)

    # construct train dataset tsv
    for run in range(1, FLAGS.number_datasets + 1):
        print('Experiment round =', run)
        s = FLAGS.mean_bagsize  # 10, 20, 50
        lines = []
        ss_dict = template(s, s*0.2, nexemplars)

        # print(ss_dict)
        # vals = np.asarray(list(ss_dict.keys()))
        # weights = np.asarray(list(ss_dict.values()))
        # weighted_avg = np.average(vals, weights=weights)
        # weighted_var = np.average((vals - weighted_avg) ** 2, weights=weights)
        # print(weighted_var, weighted_avg, sum(weights))

        for k in ss_dict:
            if k in bag_dict:
                nbags = ss_dict[k]
                assert len(bag_dict[k]) >= nbags, str(len(bag_dict[k]))+'<'+str(nbags)+' : '+str(k)
                samples = np.random.choice(bag_dict[k], size=nbags, replace=False)
                lines.extend(samples)
        np.random.shuffle(lines)
        print_stats(lines)

        output_fn = FLAGS.root_folder + FLAGS.train_prefix + str(run) + \
                    'datasize' + str(nexemplars) + 'bagsize' + str(s) + '.tsv'
        with open(output_fn, 'w', encoding='utf-8') as outf:
            outf.writelines(lines)


if __name__ == '__main__':
    app.run(main)
