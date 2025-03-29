# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

from qmnist import QMNIST
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from absl import app, flags, logging
DELIMIT = '\t'


def save_img(qdata, folder):
    data_loader = DataLoader(qdata, batch_size=None)
    count = 0
    for data in data_loader:
        pilimg, label = data
        wid = int(label[2])
        dig = int(label[0])
        did = int(label[3])
        # print(count, ':', wid, dig, did)
        filename = str(wid) + '_' + str(dig) + '_' + str(did)
        pilimg.save(folder + filename + '.png')
        count += 1


def writer_id(folder, json_file):
    fn_list = os.listdir(folder)
    wid_dig_dict = {}
    for f in fn_list:
        fn_split = f.split('_')
        wid = fn_split[0].strip()
        dig = fn_split[1].strip()
        if wid not in wid_dig_dict:
            wid_dig_dict[wid] = {}
        if dig not in wid_dig_dict[wid]:
            wid_dig_dict[wid][dig] = []
        wid_dig_dict[wid][dig] += [f]
    with open(json_file, 'w') as jsf:
        json.dump(wid_dig_dict, jsf)


def write_type1pairs(outfile, pair1, pair2map, class_label, maxna_per_digit):
    fn, dg = pair1
    pair2keys = [_x for _x in list(pair2map.keys()) if (_x!=dg and len(pair2map[_x])>0)]
    if len(pair2keys) >= 5:
        pair2keys = np.random.choice(pair2keys, size=(np.random.choice(4)+2), replace=False).tolist()
    if dg in pair2map:
        pair2keys += [dg,]
    pair2list = []
    # print(pair2keys, pair2map)
    for k in pair2keys:
        fn_list = pair2map[k]
        anc_dig = min(maxna_per_digit, len(fn_list)) if maxna_per_digit else len(fn_list)
        chosen_ancs = np.random.choice(fn_list, size=(np.random.choice(anc_dig) + 1), replace=False)
        pair2list.extend([(a, k) for a in chosen_ancs])
    assert len(pair2list) > 0, f'{pair1}, {pair2map}'
    write_line(outfile, dig_tuple=pair1, anc_tplst=pair2list, class_lbl=class_label)
    return len(pair2list)


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


def write_line(output_file, dig_tuple, anc_tplst, class_lbl):
    anc_files, anc_digit = zip(*anc_tplst)
    loclist = [i for i, c in enumerate(anc_digit) if c == dig_tuple[1]]
    output_line = dig_tuple[0] + DELIMIT + str(list(anc_files)) + DELIMIT + class_lbl + DELIMIT + str(loclist)
    output_file.write(output_line + '\n')


def write_file(json_filename, out_filename, root_path='', maxna_per_digit=None, num_rounds=1, ptype=2):
    assert ptype in [1, 2]
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
                    if ptype == 1:
                        # 1
                        pair2 = {k: v for k, v in ids_dict[w].items() if len(v) > 0}
                        if pair2:
                            linesize1 = write_type1pairs(outf, pair1=dt, pair2map=pair2, class_label='1', maxna_per_digit=maxna_per_digit)
                        else:
                            linesize1 = 0
                        # 0
                        another_w = np.random.choice([wr for wr in wrtids if (wr!=w and d in ids_dict[wr] and len(ids_dict[wr][d])>0)])
                        linesize0 = write_type1pairs(outf, pair1=dt, pair2map=ids_dict[another_w], class_label='0', maxna_per_digit=maxna_per_digit)
                    else:
                        other_wids = [wr for wr in wrtids if (wr != w and d in ids_dict[wr] and len(ids_dict[wr][d])>0)]
                        # 1
                        if len(ids_dict[w][d]) > 0:
                            writer_map1 = {w: ids_dict[w][d]}
                            other_w = np.random.choice(other_wids, size=(np.random.choice(3)+2), replace=False)
                            writer_map1.update({owid: ids_dict[owid][d] for owid in other_w})
                            linesize1 = write_type2pairs(outf, pair1=dt, pair1wid=w, pair2map=writer_map1, class_label='1', maxna_per_digit=maxna_per_digit)
                        else:
                            linesize1 = 0
                        # 0
                        all_w = np.random.choice(other_wids, size=(np.random.choice(3)+3), replace=False)
                        writer_map0 = {awid: ids_dict[awid][d] for awid in all_w}
                        linesize0 = write_type2pairs(outf, pair1=dt, pair1wid=w, pair2map=writer_map0, class_label='0', maxna_per_digit=maxna_per_digit)
                    linecount += 2
                    if max(linesize0, linesize1) > max_na:
                        max_na = max(linesize0, linesize1)
            print(linecount, ': ', max_na, '; total # writers =', len(wrtids))


root_folder = 'qmnist/'
img_folder_train = 'train60k/'
img_folder_test = 'test10k/'
temp_folder = 'tmp/'
num_ds = 3
proportion = 0.5
train_output_prefix = 'train_qmnist'
dev_output_prefix = 'dev_qmnist'
test_output_prefix = 'test_qmnist'

flags.DEFINE_bool('download_qmnist', True, 'Whether to automatically download QMNIST raw data')
flags.DEFINE_string('root_folder', root_folder, 'Folder/path as root to store output and other files')
flags.DEFINE_string('image_folder60k', img_folder_train, 'Folder/path under root to store images from MNIST-train60k')
flags.DEFINE_string('image_folder10k', img_folder_test, 'Folder/path under root to store images from MNIST-test10k')
flags.DEFINE_string('temp_folder', temp_folder, 'Folder/path under root to store temporary files during data prep')
flags.DEFINE_integer('number_datasets', num_ds,
                     'Number of round of experiement that requires construction of datasets')
flags.DEFINE_float('dev_writerprop', proportion,
                   'Proportion in number of writers for dev dataset, randomly sampled from MNIST-test10k',
                   lower_bound=0., upper_bound=1.)
flags.DEFINE_string('train_prefix', train_output_prefix, 'Prefix in the output filename of a train dataset')
flags.DEFINE_string('dev_prefix', dev_output_prefix, 'Prefix in the output filename of a dev dataset')
flags.DEFINE_string('test_prefix', test_output_prefix, 'Prefix in the output filename of a test dataset')
FLAGS = flags.FLAGS


def main(unused_argv):
    logging.set_verbosity(logging.ERROR)
    # load_qmnist
    image_folder60k = FLAGS.root_folder + FLAGS.image_folder60k
    image_folder10k = FLAGS.root_folder + FLAGS.image_folder10k
    os.makedirs(image_folder60k, exist_ok=True)
    os.makedirs(image_folder10k, exist_ok=True)
    qtrain = QMNIST(FLAGS.root_folder, what='train', compat=False, download=FLAGS.download_qmnist)
    save_img(qdata=qtrain, folder=image_folder60k)
    qtest10k = QMNIST(FLAGS.root_folder, what='test10k', compat=False, download=FLAGS.download_qmnist)
    save_img(qdata=qtest10k, folder=image_folder10k)

    # build_writer_id
    temp_folder = FLAGS.root_folder + FLAGS.temp_folder
    os.makedirs(temp_folder, exist_ok=True)
    train_json = temp_folder + 'train60k_writerid.json'
    test10k_json = temp_folder + 'test10k_writerid.json'
    writer_id(folder=image_folder60k, json_file=train_json)
    writer_id(folder=image_folder10k, json_file=test10k_json)
    # split writer_id for dev/test
    with open(test10k_json, 'r') as testf:
        test_ids = json.load(testf)
    rng = np.random.default_rng(20221231)
    rand = rng.uniform(size=len(test_ids))
    dict1 = {}
    dict2 = {}
    for i, w in enumerate(test_ids):
        if rand[i] > FLAGS.dev_writerprop:
            dict1[w] = test_ids[w]
        else:
            dict2[w] = test_ids[w]
    test_json = temp_folder + 'test_writerid.json'
    dev_json = temp_folder + 'dev_writerid.json'
    with open(dev_json, 'w') as f1:
        json.dump(dict1, f1)
    with open(test_json, 'w') as f2:
        json.dump(dict2, f2)

    # build_data
    ptype = 2
    maxna_pd = 3 if ptype == 1 else 5
    nr = 2
    # train: 21534: 25; total  # writers = 539
    # test: 2612: 19; total  # writers = 145
    # dev: 2772: 18; total  # writers = 152

    for ds in range(1, FLAGS.number_datasets + 1):
        write_file(json_filename=train_json,
                   out_filename=FLAGS.root_folder + FLAGS.train_prefix + str(ds) + '.tsv',
                   root_path=FLAGS.image_folder60k,
                   maxna_per_digit=maxna_pd,
                   num_rounds=nr,
                   ptype=ptype)
        write_file(json_filename=test_json,
                   out_filename=FLAGS.root_folder + FLAGS.test_prefix + str(ds) + '.tsv',
                   root_path=FLAGS.image_folder10k,
                   maxna_per_digit=maxna_pd,
                   num_rounds=1,
                   ptype=ptype)
        write_file(json_filename=dev_json,
                   out_filename=FLAGS.root_folder + FLAGS.dev_prefix + str(ds) + '.tsv',
                   root_path=FLAGS.image_folder10k,
                   maxna_per_digit=maxna_pd,
                   num_rounds=1,
                   ptype=ptype)


if __name__ == '__main__':
    app.run(main)
