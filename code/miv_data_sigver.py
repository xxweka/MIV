# Copyright (c) authors. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import cv2
import numpy as np
from absl import app, flags
np.random.seed()
DELIMIT = '\t'


def _to_bw(in_img, out_img, resize=None):
    img = cv2.imread(in_img, cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    # cv2.imshow('', img)
    # cv2.waitKey(0)
    # # image = Image.fromarray(img)
    # # image.show()
    bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if resize is not None:
        _bw = cv2.resize(bw, resize, interpolation=cv2.INTER_LANCZOS4)
    else:
        _bw = bw
    cv2.imwrite(out_img, _bw)


def copy_train(from_path, to_path, gf=None, cd=None):
    assert gf in ['G', 'F'] and cd in ['C', 'D']
    files = os.listdir(from_path)
    for fn in files:
        fnsplit = os.path.splitext(os.path.basename(fn))
        basefn = fnsplit[0]
        ext = fnsplit[1].upper()
        sp = basefn.split('_')
        img_id = sp[1].strip().zfill(2)
        fnid = sp[0].strip()
        if gf == 'G':
            name_id = fnid
            img_id += '_'
        else:
            name_id = fnid[-3:].strip()
            img_id += '_' + fnid[:-3].strip()
        out_fn = cd + name_id + gf + img_id + ext
        copy_from = from_path + fn

        # shutil.copy(copy_from, to_path+'query/'+out_fn)
        _to_bw(copy_from, to_path + 'query/' + out_fn, resize=IMAGE_SIZE)
        if gf == 'G':
            # shutil.copy(copy_from, to_path+'anchors/'+out_fn)
            _to_bw(copy_from, to_path + 'anchors/' + out_fn, resize=IMAGE_SIZE)


def copy_test(from_path, to_path, cd=None):
    assert cd in ['C', 'D']
    paths = os.listdir(from_path)
    for name_id in paths:
        folder = from_path+'/'+name_id+'/'
        files = os.listdir(folder)
        for f in files:
            fnsplit = os.path.splitext(os.path.basename(f))
            basefn = fnsplit[0]
            ext = fnsplit[1].upper()
            sp = basefn.split('_')
            img_id = sp[0].strip().zfill(2)
            fnid = sp[1].strip()
            # print(img_id, fnid)
            if len(fnid)>3:  # F
                img_id += '_' + fnid[:-3].strip()
                gf = 'F'
            else:
                img_id += '_'
                gf = 'G'
            out_fn = cd + name_id + gf + img_id + ext
            copy_from = folder + f
            # shutil.copy(copy_from, to_path + 'query/' + out_fn)
            _to_bw(copy_from, to_path + 'query/' + out_fn, resize=IMAGE_SIZE)
            if gf == 'G':
                # shutil.copy(copy_from, to_path + 'anchors/' + out_fn)
                _to_bw(copy_from, to_path + 'anchors/' + out_fn, resize=IMAGE_SIZE)


def writer_id(folder, json_file, check_gf=False):
    fn_list = os.listdir(folder)
    wid_dict = {}
    for f in fn_list:
        fn_split = f.split('_')[0]
        wid = fn_split[:4].strip()
        df = fn_split[4].strip()
        if check_gf:
            if wid not in wid_dict:
                wid_dict[wid] = {}
            if df not in wid_dict[wid]:
                wid_dict[wid][df] = []
            wid_dict[wid][df].append(f)
        else:
            if wid not in wid_dict:
                wid_dict[wid] = []
            wid_dict[wid].append(f)
    with open(json_file, 'w') as jsf:
        json.dump(wid_dict, jsf)


def load_wid(dict):
    cwid = []
    dwid = []
    for k in dict:
        if k.startswith('C'):
            cwid.append(k)
        else:
            dwid.append(k)
    return cwid, dwid


def write_line(output_file, qry, anc_list, anc_nlist, class_lbl, qry_root, anc_root, write_loc):
    global global_counter
    merge_loc = global_counter % len(anc_nlist)
    merge_len = len(anc_list)
    if write_loc:
        loclist = list(range(merge_loc, merge_loc+merge_len))
    else:
        loclist = []
    anchor = [anc_root + fl for fl in anc_nlist]
    anchor[merge_loc:merge_loc] = [anc_root + f for f in anc_list]
    query = qry_root + qry
    output_line = str(query) + DELIMIT + str(anchor) + DELIMIT + str(class_lbl) + DELIMIT + str(loclist)
    output_file.write(output_line + '\n')
    global_counter += 1
    return


def _widn2files(anc_widnlist, ancdict):
    global global_counter
    ret_list = []
    randnum = len(anc_widnlist)
    for an in anc_widnlist:
        anclist = ancdict[an]
        ret_list.append(anclist[(global_counter+randnum) % len(anclist)])
        if DEBUG is not None and len(ret_list)==1 and ret_list[0] == DEBUG:
            print(global_counter % len(anclist), randnum, global_counter)
        global_counter += 1
        if global_counter % 5 == 0:
            ret_list.append(anclist[(global_counter+randnum) % len(anclist)])
            global_counter += 1
    return ret_list


def _find_another(anc_widlist, idx, reference):
    _another = anc_widlist[idx % len(anc_widlist)]
    if _another.startswith(reference.split('_')[0]):
        _another = anc_widlist[(idx+1) % len(anc_widlist)]
    return _another


def write_file(anc_dict, anc_root, qry_dict, qry_root, clist, dlist, out_filename, write_loc=False):
    global global_counter
    max_bagsize = 0
    all_list = clist + dlist
    with open(out_filename, 'w') as output_file:
        for wid in all_list:
            forg = qry_dict[wid]['F']
            genu = qry_dict[wid]['G']
            size_wid = max(len(genu), len(forg))
            anc_wid = anc_dict[wid]

            # build anc_widn
            anc_widn = []
            if wid.startswith('C'):
                main_list = [cl for cl in clist if cl != wid]
                aux_list = dlist
            else:
                main_list = [dl for dl in dlist if dl != wid]
                aux_list = clist
            anc_sz = np.random.choice(4, size_wid) + 2
            for n in anc_sz:
                main_sz = n // 2 + 1
                main = np.random.choice(main_list, main_sz).tolist()
                aux = np.random.choice(aux_list, (n-main_sz)).tolist()
                anc_widn.append(main + aux)
            ian = 0
            print(wid, len(anc_widn))
            for iq in range(size_wid):
                idx_g = iq % len(genu)
                idx_f = iq % len(forg)
                curr_g = genu[idx_g]
                curr_f = forg[idx_f]

                for ia, curr_ab in enumerate(anc_wid):
                    if not curr_ab.startswith(curr_g.split('_')[0]):
                        # accepted
                        nx_awn = anc_widn[ian % len(anc_widn)]
                        curr_n = _widn2files(nx_awn, ancdict=anc_dict)
                        if DEBUG is not None and curr_n[0] == DEBUG:
                            print(curr_n, ian, nx_awn)
                        ian += 1
                        global_counter += 1
                        if (global_counter+len(curr_n)) % 5 == 0:
                            curr_aa = _find_another(anc_wid, ia + global_counter, curr_g)
                            curr_a = [curr_ab, curr_aa]
                        else:
                            curr_a = [curr_ab,]
                        if len(curr_a) + len(curr_n)>max_bagsize:
                            max_bagsize = len(curr_a) + len(curr_n)
                        write_line(output_file, curr_g, curr_a, curr_n, 1, qry_root, anc_root, write_loc)

                        # rejected
                        nx_awn = anc_widn[ian % len(anc_widn)]
                        curr_n = _widn2files(nx_awn, ancdict=anc_dict)
                        if DEBUG is not None and curr_n[0] == DEBUG:
                            print(curr_n, ian, nx_awn)
                        ian += 1
                        global_counter += 1
                        if global_counter % 10 == init_digit:
                            curr_a = []
                            if len(curr_n) > max_bagsize:
                                max_bagsize = len(curr_n)
                            write_line(output_file, curr_g, curr_a, curr_n, 0, qry_root, anc_root, write_loc)
                        else:
                            if (global_counter+len(curr_n)) % 5 == 0:
                                curr_aa = _find_another(anc_wid, ia + global_counter, curr_g)
                                curr_a = [curr_ab, curr_aa]
                            else:
                                curr_a = [curr_ab,]
                            if len(curr_a)+len(curr_n) > max_bagsize:
                                max_bagsize = len(curr_a)+len(curr_n)
                            write_line(output_file, curr_f, curr_a, curr_n, 0, qry_root, anc_root, write_loc)
    print(max_bagsize)  # 8


DEBUG = None
global_counter = np.random.choice(90)
init_digit = global_counter % 10
# print(global_counter, init_digit)
IMAGE_SIZE = (512, 256)  # w, h

flags.DEFINE_string('root_folder', 'sigcomp11/', 'Folder/path as root to store output and other files')
flags.DEFINE_string('image_raw', 'raw/', 'Folder/path under root to store raw images')
flags.DEFINE_string('temp_folder', 'tmp/', 'Folder/path under root to store temporary files during data prep')
flags.DEFINE_integer('number_datasets', 3,
                     'Number of round of experiement that requires construction of datasets')
flags.DEFINE_float('dev_writerprop', 0.1,
                   'Proportion in number of writers for dev dataset, randomly sampled from the raw data',
                   lower_bound=0., upper_bound=1.)
flags.DEFINE_float('test_writerprop', 0.1,
                   'Proportion in number of writers for test dataset, randomly sampled from the raw data',
                   lower_bound=0., upper_bound=1.)
flags.DEFINE_string('train_prefix', 'train_sigver', 'Prefix in the output filename of a train dataset')
flags.DEFINE_string('dev_prefix', 'dev_sigver', 'Prefix in the output filename of a dev dataset')
flags.DEFINE_string('test_prefix', 'test_sigver', 'Prefix in the output filename of a test dataset')
FLAGS = flags.FLAGS


def main(unused_argv):
    # images
    source_root = FLAGS.root_folder + FLAGS.image_raw
    target_root = FLAGS.root_folder
    assert os.path.exists(target_root), 'Non-existence of the root folder: ' + target_root
    assert os.path.exists(source_root), 'Non-existence of folder to store raw images: ' + source_root
    # copy and process raw images
    os.makedirs(target_root + 'query/', exist_ok=True)
    os.makedirs(target_root + 'anchors/', exist_ok=True)
    copy_train(source_root + 'Offline Genuine Chinese/', target_root, cd='C', gf='G')
    copy_train(source_root + 'Offline Forgeries Chinese/', target_root, cd='C', gf='F')
    copy_train(source_root + 'Offline Genuine Dutch/', target_root, cd='D', gf='G')
    copy_train(source_root + 'Offline Forgeries Dutch/', target_root, cd='D', gf='F')
    copy_test(source_root + 'Questioned(487)Chinese/', target_root, cd='C')
    copy_test(source_root + 'Ref(115)Chinese/', target_root, cd='C')
    copy_test(source_root + 'Questioned(1287)Dutch/', target_root, cd='D')
    copy_test(source_root + 'Ref(646)Dutch/', target_root, cd='D')

    # writer-ID json
    temp_folder = FLAGS.root_folder + FLAGS.temp_folder
    os.makedirs(temp_folder, exist_ok=True)
    json_qry = temp_folder + 'qry_writerid.json'
    json_anc = temp_folder + 'anc_writerid.json'
    writer_id(target_root + 'query/', json_qry, check_gf=True)
    writer_id(target_root + 'anchors/', json_anc, check_gf=False)
    with open(json_qry, 'r') as jq, open(json_anc, 'r') as ja:
        qdict = json.load(jq)
        adict = json.load(ja)
    c_wid, d_wid = load_wid(adict)

    # tsv
    for _r in range(1, 1 + FLAGS.number_datasets):
        split_dev, split_test = FLAGS.dev_writerprop, FLAGS.test_writerprop
        cwid_dev = np.random.choice(c_wid, int(len(c_wid)*split_dev), replace=False).tolist()
        cwid_test = np.random.choice([_x for _x in c_wid if _x not in cwid_dev], int(len(c_wid)*split_test), replace=False).tolist()
        cwid_train = [_x for _x in c_wid if _x not in cwid_dev and _x not in cwid_test]
        dwid_dev = np.random.choice(d_wid, int(len(d_wid)*split_dev), replace=False).tolist()
        dwid_test = np.random.choice([_x for _x in d_wid if _x not in dwid_dev], int(len(d_wid)*split_test), replace=False).tolist()
        dwid_train = [_x for _x in d_wid if _x not in dwid_dev and _x not in dwid_test]
        # print(cwid_train, cwid_dev, cwid_test, dwid_train, dwid_dev, dwid_test)
        write_file(anc_dict=adict, anc_root='anchors/', qry_dict=qdict, qry_root='query/',
                   clist=cwid_train, dlist=dwid_train, write_loc=False,  # True,  #
                   out_filename=target_root + FLAGS.train_prefix + str(_r) + '.tsv')
        write_file(anc_dict=adict, anc_root='anchors/', qry_dict=qdict, qry_root='query/',
                   clist=cwid_dev, dlist=dwid_dev, write_loc=False,  # True,  #
                   out_filename=target_root + FLAGS.dev_prefix + str(_r) + '.tsv')
        write_file(anc_dict=adict, anc_root='anchors/', qry_dict=qdict, qry_root='query/',
                   clist=cwid_test, dlist=dwid_test, write_loc=True,
                   out_filename=target_root + FLAGS.test_prefix + str(_r) + '.tsv')


if __name__ == '__main__':
    app.run(main)
