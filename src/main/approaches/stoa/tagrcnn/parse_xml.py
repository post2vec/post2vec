import xml.etree.ElementTree as ET
import pandas as pd
import re
import os
# from sklearn.cross_validation import train_test_split
# from ast import literal_eval


# def write_to_file(folder, post_id, text, tags):
#     with open(folder + '/50/text/' + post_id + '.txt', 'w', encoding='utf8') as f:
#         f.write(text)
#     with open(folder + '/50/tags/' + post_id + '.txt', 'w', encoding='utf8') as f:
#         f.write('\n'.join(tags))

# 建立各个数据的文件夹
# path = input('Directory: ')
# folder = re.findall(r'([a-zA-Z]+)\.', path)[0]
# if not os.path.exists(folder):
#     os.makedirs(folder + '/50')
    # os.makedirs(folder + '/1')
    # os.makedirs(folder + '/text')
    # os.makedirs(folder + '/tags')

datasets = ['askubuntu', 'codereview', 'unix', 'serverfault']
for dataset in datasets:
    path = 'community-data/' + dataset + '.stackexchange.com'
    folder = 'community-data/' + dataset + '/1'
# tags_df = pd.read_csv(folder + '/tags_df.csv', encoding='utf8', index_col=0)
    # tags_df = pd.DataFrame(columns=['TagName', 'Count'])
    # tags_tree = ET.parse(path + '\\Tags.xml')
    # for tag in tags_tree.findall('row'):
    #     tag_name = tag.get('TagName')
    #     tag_count = tag.get('Count')
    #     if tag_name is not None and tag_count is not None:
    #         tags_df = tags_df.append(
    #             {'TagName': tag_name, 'Count': int(tag_count)}, ignore_index=True)
    # tags_df.to_csv(folder + '/tags_df.csv', encoding='utf8')


    # posts_df = pd.read_csv(folder + '/1/posts_df.csv',
    #                        encoding='utf8', index_col=0)
    posts_df = pd.DataFrame(columns=['Id', 'Text', 'Tags'])
    posts_tree = ET.parse(path + '/Posts.xml')
    for i, row in enumerate(posts_tree.findall('row')):
        post_id = row.get('Id')
        body = row.get('Body')
        body = re.sub('<.+?>', '', body)
        body = re.sub('\s+', ' ', body).strip()
        title = row.get('Title')
        text = ''
        if title is not None:
            text = title
        if body is not None:
            text += (' ' + body)
        if text == '':
            continue
        tags = row.get('Tags')
        if tags is not None:
            tags = re.findall('<(.+?)>', tags)
            # tags = tags_df['TagName'][
            #     (tags_df['TagName'].isin(tags)) & (tags_df['Count'] >= 50)].tolist()
        else:
            continue
        # print('{},{},{},{}'.format(post_id, body, title, tags))
        if len(tags) != 0:
            # print(tags.tolist())
            # write_to_file(folder, post_id, text, tags)
            posts_df = posts_df.append({'Id': post_id, 'Text': text,
                                        'Tags': tags}, ignore_index=True)
            print(i)
    posts_df.to_csv(folder + '/posts_df.csv', encoding='utf8')

# posts_df['Tags'] = posts_df['Tags'].apply(
#     lambda tags: tags_df[(tags_df['TagName'].isin(literal_eval(tags)))].index.tolist())
# posts_df['Tags'] = posts_df['Tags'].apply(
#     lambda tags: tags_df[(tags_df['TagName'].isin(literal_eval(tags))) & (tags_df['Count'] >= 50)].index.tolist())
# posts_df = posts_df[posts_df['Tags'].str.len() != 0]
# X = posts_df['Text']
# y = posts_df['Tags']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=10)
# train = zip(y_train, X_train)
# train = [(tag, item[1]) for item in train for tag in item[0]]
# test = zip(y_test, X_test)
# test_for_predict = [(item[0][0], item[1]) for item in test]
# # 写入训练集
# with open(folder + '/50/train.csv', 'w', encoding='utf8') as f:
#     for (tags, text) in train:
#         # for tag in tags:
#         f.write('__label__' + str(tags) + ' ')
#         f.write(', ' + text + '\n')
# # 写入用于预测的测试集
# with open(folder + '/50/test.csv', 'w', encoding='utf8') as f:
#     for (tags, text) in test_for_predict:
#         # for tag in tags:
#         f.write('__label__' + str(tags) + ' ')
#         f.write(', ' + text + '\n')
# # 写入用于评估的测试集
# with open(folder + '/50/evaluation.csv', 'w', encoding='utf8') as f:
#     for tags in y_test:
#         s = ''
#         for tag in tags:
#             s += '__label__' + str(tag) + ','
#         f.write(s[:-1] + '\n')
