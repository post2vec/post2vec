def get_topK_lines(k, old_fpath):
    # include header
    k += 2
    with open(old_fpath) as myfile:
        topKlines = [next(myfile) for x in range(k)]
    return topKlines


def write_list_to_file(list_tmp, fpath):
    with open(fpath, 'w') as f:
        for item in list_tmp:
            f.write("%s\n" % item)
    print("Writing %s successfully!" % fpath)


def write_str_to_file(str_tmp, fpath):
    with open(fpath, "w") as text_file:
        text_file.write(str_tmp)
    print("Writing %s successfully!" % fpath)


def write_str_append_to_file(str_tmp, fpath):
    with open(fpath, "a+") as text_file:
        text_file.write(str_tmp)
    print("Writing %s successfully!" % fpath)


def read_file2dict(dict_fpath):
    dic = {}
    with open(dict_fpath) as f:
        for line in f:
            (key, val) = line.split()
            dic[int(key)] = val
    return dic


def read_file_str_list(fpath):
    with open(fpath) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


if __name__ == '__main__':
    from pathConfig import data_dir
    import os

    old_fpath = data_dir + os.sep + "tagRec/SO/SO-all-clean-with-Raretag_tmp.csv"
    new_fpath = data_dir + os.sep + "tagRec/SO/SO-all-clean-with-Raretag-100000_tmp.csv"
    # include first header line
    topK = get_topK_lines(100000, old_fpath)
    write_list_to_file(topK, new_fpath)
